#include "kernel_operator.h"
#include "fmax_sca.h"
#include "fmax_broadcast_cols.h"
#include "fmax_broadcast_3.h"
#include "fmax_broadcast.h"
#include "fmax_broadcast_small.h"
using namespace AscendC;
constexpr uint32_t BufferNum = 1;
constexpr uint32_t BufferNum_spec = 1;
template <typename T, bool isBroadcast = false>
class KernelFmax {
public:
    __aicore__ inline KernelFmax() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t smallSize, uint32_t incSize, uint32_t formerNum, TPipe *pipeIn) {
        this->pipe = pipeIn;

        uint32_t srcBeginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            srcBeginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            srcBeginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + srcBeginIndex, size);
        otherGm.SetGlobalBuffer((__gm__ T *)other + srcBeginIndex, size);
        outGm.SetGlobalBuffer((__gm__ T *)out + srcBeginIndex, size);
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bfloat16_t>::value) {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(27) * 1024 / BufferNum);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum, spaceSize * 2);
            pipe->InitBuffer(outBuf, BufferNum, spaceSize + 32);
            pipe->InitBuffer(tmpBuf, BufferNum, spaceSize * 4);
        } else if (std::is_same<T, int64_t>::value) {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(40) * 1024 / BufferNum);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum, spaceSize * 2);
            pipe->InitBuffer(outBuf, BufferNum, spaceSize + 32);
            pipe->InitBuffer(tmpBuf, BufferNum, spaceSize);
        } else {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(63) * 1024 / BufferNum);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum, spaceSize * 2);
            pipe->InitBuffer(outBuf, BufferNum, spaceSize + 32);
        }
    }
    __aicore__ inline void Init_Broadcast(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t smallSize, uint32_t incSize, uint32_t formerNum,
                                          uint32_t totalSize, uint32_t *mmInputDims, uint32_t *mmOtherDims, uint32_t *mmOutputDims, uint8_t nOutputDims,
                                          TPipe *pipeIn) {
        this->pipe = pipeIn;
        this->mmInputDims = mmInputDims;
        this->mmOtherDims = mmOtherDims;
        this->mmOutputDims = mmOutputDims;
        this->nOutputDims = nOutputDims;
        this->totalSize = totalSize;
        this->beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        this->inputSize = 1;
        this->otherSize = 1;
        for (int i = 0; i < nOutputDims; i++) {
            this->inputSize *= mmInputDims[i];
            this->otherSize *= mmOtherDims[i];
        }
        max_size_for_broadcast = 1;
        if (this->inputSize != this->totalSize) {
            max_size_for_broadcast = max(this->inputSize, max_size_for_broadcast);
        }
        if (this->otherSize != this->totalSize) {
            max_size_for_broadcast = max(this->otherSize, max_size_for_broadcast);
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input, totalSize);
        otherGm.SetGlobalBuffer((__gm__ T *)other, totalSize);
        outGm.SetGlobalBuffer((__gm__ T *)out, totalSize);
        uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(192 / 12) * 1024 / BufferNum_spec);
        this->n_elements_per_iter = spaceSize / sizeof(T);
        this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
        // 广播张量最大个数：14K。
        //  tmpBuf最小 spaceSize*8
        pipe->InitBuffer(tmpBuf, BufferNum_spec, max((uint32_t)(spaceSize * 9), (uint32_t)(max_size_for_broadcast * sizeof(T) + spaceSize * 2)));
        pipe->InitBuffer(inputBuf, BufferNum_spec, spaceSize * 2);
        pipe->InitBuffer(outBuf, BufferNum_spec, spaceSize);
    }
        // 会修改outputIdx的值
    __aicore__ inline void mapIndex(const LocalTensor<int> &inputIdx, const LocalTensor<int> &outputIdx, const LocalTensor<float> &tmpLocal, const int calCount,
                                    uint32_t *inputShape, uint32_t *outputShape, const int nDims) {
        LocalTensor<float> temp_float = tmpLocal;
        LocalTensor<float> tmpRemainingIdx_float = tmpLocal[calCount];

        LocalTensor<float> remainingIdx_float = outputIdx.template ReinterpretCast<float>();
        LocalTensor<float> inputIdx_float = inputIdx.template ReinterpretCast<float>();
        Cast(remainingIdx_float, outputIdx, RoundMode::CAST_NONE, calCount);

        Duplicate(inputIdx_float, (float)0.0, calCount);

        int stride = 1;// 输入张量的步长累积

        for (int dim = nDims - 1; dim >= 0; --dim) {// 从尾向头处理
            int intermediate = outputShape[dim]; // 将 uint32_t 转换为 int32_t
            float result = intermediate;   // 再将 int32_t 转换为 float
            Duplicate(temp_float, result, calCount);
            // 暂存一下remainingIdx_float
            Adds(tmpRemainingIdx_float, remainingIdx_float, (float)0.0, calCount);

            Div(remainingIdx_float, remainingIdx_float, temp_float, calCount);
            // 截断remainingIdx_float
            Cast(outputIdx, remainingIdx_float, RoundMode::CAST_FLOOR, calCount);
            Cast(remainingIdx_float, outputIdx, RoundMode::CAST_NONE, calCount);

            Mul(temp_float, remainingIdx_float, temp_float, calCount);

            Sub(temp_float, tmpRemainingIdx_float, temp_float, calCount);
            if (inputShape[dim] != 1) {
                Muls(temp_float, temp_float, (float)stride, calCount);
                Add(inputIdx_float, inputIdx_float, temp_float, calCount);
                stride *= inputShape[dim];
            }
        }
        Cast(inputIdx, inputIdx_float, RoundMode::CAST_RINT, calCount);
    }
    // tmpLocal大小至少为 iterSize *16 B
    // srcSize就必须小于等于 iterSize *12 B
    __aicore__ inline void CopyInBroadcast(const GlobalTensor<T> &srcGm, const LocalTensor<T> &inputLocal, const LocalTensor<uint8_t> &tmpLocal,
                                           uint32_t offset, uint32_t iterSize, uint32_t *inputShape, uint32_t *outputShape, const int nDims,
                                           const uint32_t srcSize) {
        LocalTensor<int> outputIdxTensor = (tmpLocal[iterSize * 8]).template ReinterpretCast<int>();
        // 不能被修改
        LocalTensor<int> inputIdxTensor =
            (tmpLocal[max(((uint32_t)(max_size_for_broadcast * sizeof(T) + 31u) / 32u * 32u), (uint32_t)(iterSize * 12u))]).template ReinterpretCast<int>();

        CreateVecIndex(outputIdxTensor, (int)offset, iterSize);
        mapIndex(inputIdxTensor, outputIdxTensor, tmpLocal.template ReinterpretCast<float>(), iterSize, inputShape, outputShape, nDims);
        // srcLocal最多分配 iterSize *12 B
        LocalTensor<T> srcLocal = tmpLocal.template ReinterpretCast<T>();

        uint16_t blockCount = 1;
        uint32_t blockLen = srcSize * sizeof(T);
        DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        int32_t eventIDS_V_MTE2_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        SetFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(eventIDS_V_MTE2_0);

        DataCopyPad(srcLocal, srcGm, copyParams, padParams);

        int32_t eventIDS_MTE2_V_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        SetFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDS_MTE2_V_0);

        if constexpr (std::is_same<DTYPE_INPUT, float>::value || std::is_same<DTYPE_INPUT, int32_t>::value) {
            ShiftLeft(inputIdxTensor, inputIdxTensor, 2, iterSize);
        } else if constexpr (std::is_same<DTYPE_INPUT, int16_t>::value || std::is_same<DTYPE_INPUT, half>::value) {
            ShiftLeft(inputIdxTensor, inputIdxTensor, 1, iterSize);
        } else if constexpr (std::is_same<DTYPE_INPUT, bool>::value || std::is_same<DTYPE_INPUT, int8_t>::value || std::is_same<DTYPE_INPUT, uint8_t>::value) {
            ;
        } else if constexpr (std::is_same<DTYPE_INPUT, int64_t>::value) {
            ShiftLeft(inputIdxTensor, inputIdxTensor, 3, iterSize);
        }
        Gather(inputLocal, srcLocal, inputIdxTensor.template ReinterpretCast<uint32_t>(), 0, iterSize);
    }
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {
        LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
        LocalTensor<T> otherLocal = inputLocal[iterSize];

        // 从全局内存读取数据到本地内存
        if constexpr (isBroadcast) {
            LocalTensor<uint8_t> tmpLocal = tmpBuf.AllocTensor<uint8_t>();
            if (inputSize != totalSize) {
                CopyInBroadcast(inputGm, inputLocal, tmpLocal, offset, iterSize, mmInputDims, mmOutputDims, nOutputDims, inputSize);
            } else {
                DataCopy(inputLocal, inputGm[offset], iterSize);
            }
            if (otherSize != totalSize) {
                CopyInBroadcast(otherGm, otherLocal, tmpLocal, offset, iterSize, mmOtherDims, mmOutputDims, nOutputDims, otherSize);
            } else {
                DataCopy(otherLocal, otherGm[offset], iterSize);
            }
            tmpBuf.FreeTensor<uint8_t>(tmpLocal);
        } else {
            DataCopy(inputLocal, inputGm[offset], iterSize);
            DataCopy(otherLocal, otherGm[offset], iterSize);
        }
        inputBuf.EnQue<T>(inputLocal);
        inputLocal = inputBuf.DeQue<T>();

        LocalTensor<T> outLocal = outBuf.AllocTensor<T>();
        if constexpr (isBroadcast) {
            PipeBarrier<PIPE_ALL>();
        }
        // 计算
        if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
                      std::is_same<T, bool>::value) {
            Compute_iter(outLocal, inputLocal, otherLocal, iterSize);
        } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            LocalTensor<half> inputLocal_fp16 = tmpBuf.AllocTensor<half>();
            LocalTensor<half> otherLocal_fp16 = inputLocal_fp16[iterSize];
            LocalTensor<half> outLocal_fp16 = inputLocal.template ReinterpretCast<half>();
            Cast(inputLocal_fp16, inputLocal, RoundMode::CAST_NONE, iterSize);
            Cast(otherLocal_fp16, otherLocal, RoundMode::CAST_NONE, iterSize);
            Compute_iter(outLocal_fp16, inputLocal_fp16, otherLocal_fp16, iterSize);
            Cast(outLocal, outLocal_fp16, RoundMode::CAST_NONE, iterSize);
            tmpBuf.FreeTensor<half>(inputLocal_fp16);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            LocalTensor<float> inputLocal_fp32 = tmpBuf.AllocTensor<float>();
            LocalTensor<float> otherLocal_fp32 = inputLocal_fp32[iterSize];
            LocalTensor<float> outLocal_fp32 = inputLocal.template ReinterpretCast<float>();
            Cast(inputLocal_fp32, inputLocal, RoundMode::CAST_NONE, iterSize);
            Cast(otherLocal_fp32, otherLocal, RoundMode::CAST_NONE, iterSize);
            Compute_iter(outLocal_fp32, inputLocal_fp32, otherLocal_fp32, iterSize);
            Cast(outLocal, outLocal_fp32, RoundMode::CAST_RINT, iterSize);
            tmpBuf.FreeTensor<float>(inputLocal_fp32);
        } else { // int64
            LocalTensor<uint8_t> tmpLocal = tmpBuf.AllocTensor<uint8_t>();
            Compute_int64_iter(outLocal.template ReinterpretCast<int64_t>(), inputLocal.template ReinterpretCast<int64_t>(),
                               otherLocal.template ReinterpretCast<int64_t>(), tmpLocal, iterSize);
            tmpBuf.FreeTensor<uint8_t>(tmpLocal);
        }
        inputBuf.FreeTensor<T>(inputLocal);

        outBuf.EnQue<T>(outLocal);
        outLocal = outBuf.DeQue<T>();
        // 将结果写回全局内存
        DataCopy(outGm[offset], outLocal, iterSize);
        outBuf.FreeTensor<T>(outLocal);
    }
    template <typename U>
    __aicore__ inline void Compute_iter(const LocalTensor<U> &outLocal, const LocalTensor<U> &inputLocal, const LocalTensor<U> &otherLocal, uint32_t iterSize) {
        if constexpr (std::is_same<U, bool>::value) {
            LocalTensor<uint32_t> inputLocal_uint16 = inputLocal.template ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> otherLocal_uint16 = otherLocal.template ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> outLocal_uint16 = outLocal.template ReinterpretCast<uint32_t>();
            Or(outLocal_uint16, inputLocal_uint16, otherLocal_uint16, iterSize / 2);
        } else {
            Max(outLocal, inputLocal, otherLocal, iterSize);
        }
    }
    // tmpLocal大小至少为 iterSize *8 B
    __aicore__ inline void Compute_int64_iter(const LocalTensor<int64_t> &outLocal, const LocalTensor<int64_t> &inputLocal,
                                              const LocalTensor<int64_t> &otherLocal, const LocalTensor<uint8_t> &tmpLocal, uint32_t iterSize) {
        LocalTensor<int32_t> inputLocal_int32 = inputLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> otherLocal_int32 = otherLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> outLocal_int32 = outLocal.template ReinterpretCast<int32_t>();
        // 用掉了 iterSize*8 B的临时空间
        LocalTensor<int32_t> inputLocal_hi = tmpLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> inputLocal_lo = inputLocal_hi[iterSize];
        LocalTensor<int32_t> otherLocal_lo = inputLocal_int32;// 复用inputLocal_int32的空间
        LocalTensor<int32_t> otherLocal_hi = otherLocal_lo[iterSize];

        // 需要iterSize bit的临时空间,即iterSize/8 byte
        LocalTensor<uint8_t> high_equal = otherLocal_int32.template ReinterpretCast<uint8_t>();// 复用otherLocal_int32的空间
        LocalTensor<uint8_t> max_is_x1 = high_equal[iterSize / 8];
        LocalTensor<uint8_t> mask = max_is_x1[iterSize / 8];
        uint64_t rsvdCnt = 0;
        uint8_t repeatTimes = (uint8_t)((iterSize + 31) / 32);
        GatherMask(inputLocal_hi, inputLocal_int32, (uint8_t)2, false, 0, {1, repeatTimes, 8, 0}, rsvdCnt);
        GatherMask(inputLocal_lo, inputLocal_int32, (uint8_t)1, false, 0, {1, repeatTimes, 8, 0}, rsvdCnt);

        GatherMask(otherLocal_hi, otherLocal_int32, (uint8_t)2, false, 0, {1, repeatTimes, 8, 0}, rsvdCnt);
        GatherMask(otherLocal_lo, otherLocal_int32, (uint8_t)1, false, 0, {1, repeatTimes, 8, 0}, rsvdCnt);

        Compare(high_equal, inputLocal_hi, otherLocal_hi, CMPMODE::EQ, iterSize);
        Max(outLocal_int32, inputLocal_hi, otherLocal_hi, iterSize);
        Compare(max_is_x1, inputLocal_hi, outLocal_int32, CMPMODE::EQ, iterSize);
        Not(mask.template ReinterpretCast<uint16_t>(), high_equal.template ReinterpretCast<uint16_t>(), iterSize / 16);
        // mask代表x1_high > x2_high
        And(mask.template ReinterpretCast<uint16_t>(), mask.template ReinterpretCast<uint16_t>(), max_is_x1.template ReinterpretCast<uint16_t>(),
            iterSize / 16);

        Adds(inputLocal_lo, inputLocal_lo, (int32_t)0x80000000, iterSize);
        Adds(otherLocal_lo, otherLocal_lo, (int32_t)0x80000000, iterSize);

        Max(outLocal_int32, inputLocal_lo, otherLocal_lo, iterSize);
        Compare(max_is_x1, inputLocal_lo, outLocal_int32, CMPMODE::EQ, iterSize);
        // 只在高位相等时有效
        And(max_is_x1.template ReinterpretCast<uint16_t>(), max_is_x1.template ReinterpretCast<uint16_t>(), high_equal.template ReinterpretCast<uint16_t>(),
            iterSize / 16);
        Or(mask.template ReinterpretCast<uint16_t>(), mask.template ReinterpretCast<uint16_t>(), max_is_x1.template ReinterpretCast<uint16_t>(), iterSize / 16);

        Adds(inputLocal_lo, inputLocal_lo, (int32_t)0x80000000, iterSize);
        Adds(otherLocal_lo, otherLocal_lo, (int32_t)0x80000000, iterSize);

        // mask代表x1 > x2
        Select(inputLocal_int32.template ReinterpretCast<float>(), mask, inputLocal_lo.template ReinterpretCast<float>(),
               otherLocal_lo.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Select(inputLocal_int32[iterSize].template ReinterpretCast<float>(), mask, inputLocal_hi.template ReinterpretCast<float>(),
               otherLocal_hi.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        // 因为没有scatter，就只能生成index，然后用gather
        // 需要生成[0,0,1,1,....,N-1,N-1],考虑使用Broadcast实现N,1->N,2
        // 再用Adds 和mask实现生成[0,0+N,1,1+N,....,N-1,N-1+N]
        // 再用Gather实现合并成int64
        LocalTensor<uint8_t> sharedTmpBuffer = outLocal_int32.template ReinterpretCast<uint8_t>();// 这个空间够用吗？
        LocalTensor<int32_t> indexLocal = tmpLocal.template ReinterpretCast<int32_t>();
        LocalTensor<float> indexLocal_fp32 = indexLocal.template ReinterpretCast<float>();
        LocalTensor<int32_t> indexLocal_broadcast = otherLocal.template ReinterpretCast<int32_t>();
        LocalTensor<float> indexLocal_broadcast_fp32 = indexLocal_broadcast.template ReinterpretCast<float>();
        CreateVecIndex(indexLocal, 0, iterSize);
        // 需要临时空间
        uint32_t broadcastShapeIn[2] = {iterSize, 1};
        uint32_t broadcastShapeOut[2] = {iterSize, 2};
        Broadcast<float, 2, 1>(indexLocal_broadcast_fp32, indexLocal_fp32, broadcastShapeOut, broadcastShapeIn, sharedTmpBuffer);
        uint64_t mask_broadcast[2] = {0xAAAAAAAAAAAAAAAA, 0};
        Adds(indexLocal_broadcast, indexLocal_broadcast, (int32_t)iterSize, mask_broadcast, iterSize * sizeof(int64_t) / 256, {1, 1, 8, 8});
        // 按字节偏移，不要忘记计算个数乘以4
        ShiftLeft(indexLocal_broadcast, indexLocal_broadcast, 2, iterSize * 2);
        Gather(outLocal_int32, inputLocal_int32, indexLocal_broadcast.template ReinterpretCast<uint32_t>(), 0, iterSize * 2);
    }
    __aicore__ inline void Process() {
        uint32_t blockBeginIndex = 0;
        for (uint32_t i = 0; i < smallLoopTimes - 1; ++i) {
            if constexpr (isBroadcast) {
                Process_iter(blockBeginIndex + beginIndex, n_elements_per_iter);
            } else {
                Process_iter(blockBeginIndex, n_elements_per_iter);
            }
            blockBeginIndex += n_elements_per_iter;
        }
        uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex);
        if constexpr (isBroadcast) {
            Process_iter(blockBeginIndex + beginIndex, iterSize);
        } else {
            Process_iter(blockBeginIndex, iterSize);
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<T> outGm;
    TQue<QuePosition::VECIN, 1> inputBuf;
    // TQue<QuePosition::VECIN, 1> otherBuf;
    TQue<QuePosition::VECOUT, 1> outBuf;
    TQue<QuePosition::VECCALC, 1> tmpBuf;
    uint32_t size;
    uint32_t smallLoopTimes;
    uint32_t n_elements_per_iter;

    uint32_t totalSize;
    uint32_t inputSize;
    uint32_t otherSize;

    uint32_t *mmInputDims;
    uint32_t *mmOtherDims;
    uint32_t *mmOutputDims;
    int nOutputDims;
    uint32_t beginIndex;
    uint32_t max_size_for_broadcast;
};
extern "C" __global__ __aicore__ void fmax(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelFmax<DTYPE_INPUT, false> op;
        op.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelFmax<DTYPE_INPUT, true> op;
        op.Init_Broadcast(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims,
                          tiling_data.mmOtherDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelFmax_sca<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims,
                tiling_data.mmOtherDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelFmax_broadcast<DTYPE_INPUT> op;
        if (tiling_data.mmInputDims[0] == 1) {// input是需要广播的张量,调换一下位置
            op.Init(other, input, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmOtherDims,
                    tiling_data.mmInputDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        } else {
            op.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims,
                    tiling_data.mmOtherDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        }
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KernelFmax_broadcast_cols<DTYPE_INPUT> op;
        if (tiling_data.mmInputDims[tiling_data.nOutputDims - 1] == 1) {// input是需要广播的张量,调换一下位置
            op.Init(other, input, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmOtherDims,
                    tiling_data.mmInputDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        } else {
            op.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims,
                    tiling_data.mmOtherDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        }
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        KernelFmax_broadcast_3<DTYPE_INPUT> op;
        if (tiling_data.mmInputDims[1] == 1) {// input是需要广播的张量,调换一下位置
            op.Init(other, input, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmOtherDims,
                    tiling_data.mmInputDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        } else {
            op.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims,
                    tiling_data.mmOtherDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        }
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        KernelFmax_broadcast_small<DTYPE_INPUT> op;
        if (tiling_data.mmInputDims[1] == 1) {// input是需要广播的张量,调换一下位置
            op.Init(other, input, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmOtherDims,
                    tiling_data.mmInputDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        } else {
            op.Init(input, other, out, tiling_data.smallSize, tiling_data.incSize, tiling_data.formerNum, tiling_data.totalSize, tiling_data.mmInputDims,
                    tiling_data.mmOtherDims, tiling_data.mmOutputDims, tiling_data.nOutputDims, &pipe);
        }
        op.Process();
    }
}
