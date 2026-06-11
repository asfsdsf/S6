#include "fmax_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    FmaxTilingData tiling;
    const auto shape0 = context->GetInputTensor(0)->GetOriginShape();
    const auto shape1 = context->GetInputTensor(1)->GetOriginShape();

    uint8_t nOutputDims = std::max(shape0.GetDimNum(), shape1.GetDimNum());
    uint32_t arr0[8];
    for (int i = 0; i < nOutputDims; i++) {
        if (shape0.GetDimNum() + i >= nOutputDims) {
            arr0[i] = shape0.GetDim(shape0.GetDimNum() - nOutputDims + i);
        } else {
            arr0[i] = 1;
        }
    }
    uint32_t arr1[8];
    for (int i = 0; i < nOutputDims; i++) {
        if (shape1.GetDimNum() + i >= nOutputDims) {
            arr1[i] = shape1.GetDim(shape1.GetDimNum() - nOutputDims + i);
        } else {
            arr1[i] = 1;
        }
    }
    uint32_t tmp0[8], tmp1[8];
    int wp = 0;                       // 写入指针（最右开始）
    auto flush = [&](uint32_t& p0, uint32_t& p1, int& cnt) {
        if (cnt == 0) return;
        tmp0[wp] = static_cast<uint32_t>(p0);
        tmp1[wp] = static_cast<uint32_t>(p1);
        ++wp;
        cnt = 0;
        p0 = p1 = 1;
    };
    uint32_t prod0 = 1, prod1 = 1;
    int cnt = 0;

    for (int rp = nOutputDims - 1; rp >= 0; --rp) {  // 从右往左读
        if (arr0[rp] != 1 && arr1[rp] != 1) {        // 都可合并
            prod0 *= arr0[rp];
            prod1 *= arr1[rp];
            ++cnt;
        } else {                                     // 遇到1 →  flush
            flush(prod0, prod1, cnt);
            tmp0[wp] = arr0[rp];   // 把当前1或广播维原样写入
            tmp1[wp] = arr1[rp];
            ++wp;
        }
    }
    flush(prod0, prod1, cnt);   // 别漏最后一段

    nOutputDims = wp;
    for (int i = 0; i < nOutputDims; ++i) {
        arr0[i] = tmp0[nOutputDims - 1 - i];
        arr1[i] = tmp1[nOutputDims - 1 - i];
        // std::cout << "arr0[" << i << "]=" << arr0[i] << ", arr1[" << i << "]=" << arr1[i] << std::endl;
    }

    uint32_t arr[8];
    uint32_t outputSize = 1;
    bool isBroadcast = false;
    int need_broadcast_dims = 0;
    for (int i = 0; i < nOutputDims; i++) {
        if (arr0[i] != arr1[i]) {
            isBroadcast = true;
            need_broadcast_dims++;
        }
        arr[i] = std::max(arr0[i], arr1[i]);
        outputSize *= arr[i];
    }
    tiling.set_mmInputDims(arr0);
    tiling.set_mmOtherDims(arr1);
    tiling.set_totalSize(outputSize);
    tiling.set_nOutputDims(nOutputDims);
    tiling.set_mmOutputDims(arr);

    uint32_t size = outputSize;
    uint32_t aivNum = 40; // Ascend910B
    auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
    int DataTypeSize = 0;
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        DataTypeSize = 4;
    } else if (dt == ge::DT_BF16 || dt == ge::DT_FLOAT16 || dt == ge::DT_INT16) {
        DataTypeSize = 2;
    } else if (dt == ge::DT_BOOL || dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        DataTypeSize = 1;
    } else if (dt == ge::DT_INT64) {
        DataTypeSize = 8;
    }
    // blockSize以byte为单位
    uint32_t blockSize = 512;
    int max_size_per_iter = DataTypeSize == 4 ? 15 * 1024 : 13 * 1024; // float类型每次迭代最大处理16K元素，其他类型8K元素
    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8 || dt == ge::DT_BF16) {
        max_size_per_iter = 27 * 1024 / DataTypeSize;
    } else if (dt == ge::DT_INT64) {
        max_size_per_iter = 40 * 1024 / DataTypeSize;
    } else {
        max_size_per_iter = 63 * 1024 / DataTypeSize;
    }

    bool use_sca = true;
    if (!isBroadcast) {
        // 非广播模式
        context->SetTilingKey(1);
    } else {
        if (nOutputDims == 2) {
            int H = arr[0];
            int W = arr[1];
            if (arr0[0] == 1 || arr1[0] == 1) {// 行广播
                size = W;// 把W划分了
                use_sca = false;
                if (W <= 40 * 4 * 1024) {
                    context->SetTilingKey(7);
                } else {
                    context->SetTilingKey(4);
                }
            } else if (DataTypeSize != 8) {// 列广播,先不考虑int64
                use_sca = false;
                size = H;// 把H划分了
                // 隐藏bug预定
                blockSize = DataTypeSize;
                context->SetTilingKey(5);
            }
        } else if (nOutputDims == 3) {
            int D = arr[0];
            int H = arr[1];
            int W = arr[2];
            if (W < max_size_per_iter && DataTypeSize != 8) {// 先考虑W很小的情况,且不考虑int64
                use_sca = false;
                size = D;// 把D划分了
                // 隐藏bug预定
                blockSize = DataTypeSize;
                context->SetTilingKey(6);
            }
        }
        if (use_sca) {
            context->SetTilingKey(3);
        }
    }

    uint32_t blockNum = (size * DataTypeSize + blockSize - 1) / blockSize;
    aivNum = std::min(aivNum, blockNum);

    uint32_t smallSize = blockNum / aivNum * blockSize / DataTypeSize;
    uint32_t incSize = blockSize / DataTypeSize;
    uint16_t formerNum = blockNum % aivNum;
    // std::cout << "aivNum=" << aivNum << ", blockNum=" << blockNum << ", smallSize=" << smallSize << ", incSize=" << incSize << ", formerNum=" << formerNum
    //           << std::endl;
    tiling.set_smallSize(smallSize);
    tiling.set_incSize(incSize);
    tiling.set_formerNum(formerNum);
    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Fmax : public OpDef {
public:
    explicit Fmax(const char* name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64, ge::DT_INT16, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Fmax);
} // namespace ops
