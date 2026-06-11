#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t BufferNum_broadcast = 1;
// other张量始终是需要广播的张量,只考虑了二维行广播
template <typename T>
class KernelFmax_broadcast {
public:
    __aicore__ inline KernelFmax_broadcast() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, uint32_t smallSize, uint32_t incSize, uint32_t formerNum, uint32_t totalSize,
                                uint32_t *mmInputDims, uint32_t *mmOtherDims, uint32_t *mmOutputDims, uint8_t nOutputDims, TPipe *pipeIn) {
        this->pipe = pipeIn;
        this->mmInputDims = mmInputDims;
        this->mmOtherDims = mmOtherDims;
        this->mmOutputDims = mmOutputDims;
        this->nOutputDims = nOutputDims;
        this->totalSize = totalSize;

        if (nOutputDims == 2) {// 暂时先考虑二维情况
            this->D = 1;
            this->H = mmOutputDims[0];
            this->W = mmOutputDims[1];
        } else if (nOutputDims == 3) {
            this->D = mmOutputDims[0];
            this->H = mmOutputDims[1];
            this->W = mmOutputDims[2];
        }

        this->beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            this->size = smallSize + incSize;
            beginIndex = this->size * GetBlockIdx();
        } else {
            this->size = smallSize;
            beginIndex = this->size * GetBlockIdx() + formerNum * incSize;
        }
        if (beginIndex + size > W) {// 最后一块
            this->size = W - beginIndex;
        }
        inputGm.SetGlobalBuffer((__gm__ T *)input + beginIndex, totalSize);
        otherGm.SetGlobalBuffer((__gm__ T *)other + beginIndex, totalSize);
        outGm.SetGlobalBuffer((__gm__ T *)out + beginIndex, totalSize);

        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, bfloat16_t>::value) {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(31) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            // pipe->InitBuffer(otherBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(tmpBuf, BufferNum_sca, spaceSize * 4);
        } else if (std::is_same<T, int64_t>::value) {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(40) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            pipe->InitBuffer(otherBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(tmpBuf, BufferNum_sca, spaceSize);
        } else {
            uint32_t spaceSize = min((uint32_t)(size * sizeof(T)), (uint32_t)(63) * 1024 / BufferNum_sca);
            this->n_elements_per_iter = spaceSize / sizeof(T);
            this->smallLoopTimes = (size + n_elements_per_iter - 1) / n_elements_per_iter;
            pipe->InitBuffer(inputBuf, BufferNum_sca, spaceSize);
            pipe->InitBuffer(otherBuf, BufferNum_sca, spaceSize + 32);
            pipe->InitBuffer(outBuf, BufferNum_sca, spaceSize + 32);
        }
    }
    __aicore__ inline void Process_iter(uint32_t offset, uint32_t iterSize) {
        uint16_t blockCount = 1;
        uint32_t blockLen = iterSize * sizeof(T);
        DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> otherLocal = inputBuf.AllocTensor<T>();
        DataCopyPad(otherLocal, otherGm[offset], copyParams, padParams);
        inputBuf.EnQue<T>(otherLocal);
        otherLocal = inputBuf.DeQue<T>();
        LocalTensor<float> inputLocal_fp32 = tmpBuf.AllocTensor<float>();
        LocalTensor<float> otherLocal_fp32 = inputLocal_fp32[(iterSize + 7) / 8 * 8];
        if constexpr (std::is_same<T, bfloat16_t>::value)
            Cast(otherLocal_fp32, otherLocal, RoundMode::CAST_NONE, iterSize);
        uint32_t half_iterSize = iterSize / 2 / 16 * 16;
        inputBuf.FreeTensor<T>(otherLocal);
        for (int32_t i = 0; i < H; i++) {
            // printf("0 otherLocal[0]=%f, otherLocal[1]=%f, otherLocal[2]=%f, otherLocal[3]=%f\n", otherLocal(0), otherLocal(1), otherLocal(2), otherLocal(3));
            LocalTensor<T> inputLocal = inputBuf.AllocTensor<T>();
            uint32_t outputIdx = i * W + offset;
            DataCopyPad(inputLocal, inputGm[outputIdx], copyParams, padParams);
            // printf("0 inputLocal[0]=%f, inputLocal[1]=%f, inputLocal[2]=%f, inputLocal[3]=%f\n", inputLocal(0), inputLocal(1), inputLocal(2), inputLocal(3));

            inputBuf.EnQue<T>(inputLocal);
            inputLocal = inputBuf.DeQue<T>();
            LocalTensor<T> outLocal = outBuf.AllocTensor<T>();

            // 计算
            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
                          std::is_same<T, bool>::value) {
                Compute_iter(outLocal, inputLocal, otherLocal, iterSize);
            } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                LocalTensor<half> inputLocal_fp16 = tmpBuf.AllocTensor<half>();
                LocalTensor<half> otherLocal_fp16 = inputLocal_fp16[(iterSize + 15) / 16 * 16];
                LocalTensor<half> outLocal_fp16 = inputLocal.template ReinterpretCast<half>();
                Cast(inputLocal_fp16, inputLocal, RoundMode::CAST_NONE, iterSize);
                Cast(otherLocal_fp16, otherLocal, RoundMode::CAST_NONE, iterSize);
                Compute_iter(outLocal_fp16, inputLocal_fp16, otherLocal_fp16, half_iterSize);
                Cast(outLocal, outLocal_fp16, RoundMode::CAST_NONE, half_iterSize);
                Compute_iter(outLocal_fp16, inputLocal_fp16[half_iterSize], otherLocal_fp16[half_iterSize], iterSize - half_iterSize);
                Cast(outLocal[half_iterSize], outLocal_fp16, RoundMode::CAST_NONE, iterSize - half_iterSize);
                tmpBuf.FreeTensor<half>(inputLocal_fp16);
            } else if constexpr (std::is_same<T, bfloat16_t>::value) {
                LocalTensor<float> outLocal_fp32 = inputLocal.template ReinterpretCast<float>();
                Cast(inputLocal_fp32, inputLocal, RoundMode::CAST_NONE, iterSize);
                Compute_iter(outLocal_fp32, inputLocal_fp32, otherLocal_fp32, half_iterSize);
                Cast(outLocal, outLocal_fp32, RoundMode::CAST_RINT, half_iterSize);
                Compute_iter(outLocal_fp32, inputLocal_fp32[half_iterSize], otherLocal_fp32[half_iterSize], iterSize - half_iterSize);
                Cast(outLocal[half_iterSize], outLocal_fp32, RoundMode::CAST_RINT, iterSize - half_iterSize);
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
            DataCopyExtParams storeParams{blockCount, blockLen, 0, 0, 0};
            // printf("5 outLocal[0]=%f, outLocal[1]=%f, outLocal[2]=%f, outLocal[3]=%f\n", outLocal(0), outLocal(1), outLocal(2), outLocal(3));
            DataCopyPad(outGm[outputIdx], outLocal, storeParams);
            outBuf.FreeTensor<T>(outLocal);
        }
        tmpBuf.FreeTensor<float>(inputLocal_fp32);
    }
    // otherLocal不会被影响
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
        LocalTensor<int32_t> otherLocal_lo = inputLocal_int32; // 复用inputLocal_int32的空间
        LocalTensor<int32_t> otherLocal_hi = otherLocal_lo[iterSize];

        // 需要iterSize bit的临时空间,即iterSize/8 byte
        LocalTensor<uint8_t> high_equal = otherLocal_int32.template ReinterpretCast<uint8_t>(); // 复用otherLocal_int32的空间
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
        LocalTensor<uint8_t> sharedTmpBuffer = outLocal_int32.template ReinterpretCast<uint8_t>(); // 这个空间够用吗？
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
            Process_iter(blockBeginIndex, n_elements_per_iter);
            blockBeginIndex += n_elements_per_iter;
        }
        uint32_t iterSize = min(n_elements_per_iter, size - blockBeginIndex);
        Process_iter(blockBeginIndex, iterSize);
    }

private:
    TPipe *pipe;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> otherGm;
    GlobalTensor<T> outGm;
    TQue<QuePosition::VECIN, 1> inputBuf;
    TQue<QuePosition::VECIN, 1> otherBuf;
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
    int D;
    int H;
    int W;
};
