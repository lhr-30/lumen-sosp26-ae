#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "IR/builtin_module.h"
#include "IR/constant_folder.h"
#include "IR/generator_context.h"
#include "IR/mlir_generator_impl.h"
#include "IR/named_module.h"
#include "IR/type_system.h"
#include "Utils/assert.h"
#include "Utils/embedded_filesystem_view.h"
#include "intrinsic_support.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/NVGPU/IR/NVGPUDialect.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <string_view>
#include <utility>

extern "C" const unsigned char _binary_nvvm_intrinsics_mlirbc_start[];
extern "C" const unsigned char _binary_nvvm_intrinsics_mlirbc_end[];

namespace causalflow::avelang::ir {

using namespace mlir;
using namespace mlir::NVVM;
using namespace causalflow::avelang::dialect;
namespace cf = causalflow::avelang::dialect;

namespace {

enum class CpAsyncBulkIntrinsicKind {
    CommitGroup,
    GlobalSharedCTA,
    Prefetch,
    SharedClusterGlobal,
    SharedClusterSharedCTA,
    TensorGlobalSharedCTA,
    TensorPrefetch,
    TensorReduce,
    TensorSharedClusterGlobal,
    WaitGroup,
};

static llvm::StringRef GetNvvmIntrinsicLibrary() {
    auto *start =
        reinterpret_cast<const char *>(_binary_nvvm_intrinsics_mlirbc_start);
    auto *end =
        reinterpret_cast<const char *>(_binary_nvvm_intrinsics_mlirbc_end);
    return {start, static_cast<size_t>(end - start)};
}

constexpr llvm::StringRef kNvvmIntrinsicLibraryName = "nvvm_intrinsics.mlirbc";
constexpr llvm::StringRef kNvvmIntrinsicLibraryTag =
    "embedded:nvvm_intrinsics.mlirbc";

static void emitInlinePtxVoid(mlir::OpBuilder &builder, mlir::Location loc,
                              llvm::StringRef asmString,
                              llvm::StringRef constraints = "") {
    mlir::LLVM::InlineAsmOp::create(
        builder, loc, mlir::TypeRange{}, mlir::ValueRange{}, asmString,
        constraints,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
}

static std::optional<mlir::nvgpu::TensorMapSwizzleKind>
getTensorMapSwizzleKind(int64_t swizzle) {
    switch (swizzle) {
    case 0:
        return mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_NONE;
    case 1:
        return mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_32B;
    case 2:
        return mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_64B;
    case 3:
        return mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_128B;
    default:
        return std::nullopt;
    }
}

static bool extractConstantTupleValues(mlir::Value tupleValue,
                                       llvm::SmallVectorImpl<int64_t> &values) {
    if (auto tupleOp = tupleValue.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            if (!extractConstantTupleValues(elem, values)) {
                return false;
            }
        }
        return true;
    }

    auto value = ConstantFolder::FoldIntValue(tupleValue);
    if (!value) {
        return false;
    }
    values.push_back(*value);
    return true;
}

static bool extractTupleValues(mlir::Value tupleValue,
                               llvm::SmallVectorImpl<mlir::Value> &values) {
    if (auto tupleOp = tupleValue.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            if (!extractTupleValues(elem, values)) {
                return false;
            }
        }
        return true;
    }

    if (!tupleValue) {
        return false;
    }
    values.push_back(tupleValue);
    return true;
}

static bool isMemRefLike(mlir::Type type) {
    return mlir::isa<cf::MemRefType, mlir::MemRefType>(type);
}

static mlir::Value createDefaultIndex(mlir::OpBuilder &builder,
                                      mlir::Location location) {
    return mlir::arith::ConstantIndexOp::create(builder, location, 0)
        .getResult();
}

static mlir::Value castIntegerTo(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Value value,
                                 mlir::IntegerType targetType) {
    if (!value) {
        return {};
    }
    if (value.getType() == targetType) {
        return value;
    }
    if (value.getType().isIndex()) {
        return mlir::arith::IndexCastOp::create(builder, location, targetType,
                                                value)
            .getResult();
    }
    auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
    if (!intType) {
        return {};
    }
    if (intType.getWidth() == targetType.getWidth()) {
        return value;
    }
    if (intType.getWidth() < targetType.getWidth()) {
        return mlir::arith::ExtUIOp::create(builder, location, targetType,
                                            value)
            .getResult();
    }
    return mlir::arith::TruncIOp::create(builder, location, targetType, value)
        .getResult();
}

static mlir::Value castToIndex(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value value) {
    if (!value) {
        return {};
    }
    if (value.getType().isIndex()) {
        return value;
    }
    if (!value.getType().isIntOrIndex()) {
        return {};
    }
    return mlir::arith::IndexCastOp::create(builder, location,
                                            builder.getIndexType(), value)
        .getResult();
}

static mlir::LLVM::LLVMPointerType
getNvvmPointerType(mlir::OpBuilder &builder,
                   mlir::NVVM::NVVMMemorySpace memorySpace) {
    return mlir::LLVM::LLVMPointerType::get(builder.getContext(),
                                            static_cast<unsigned>(memorySpace));
}

static mlir::Value castToPointer(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Value value,
                                 mlir::LLVM::LLVMPointerType pointerType) {
    if (!value) {
        return {};
    }
    if (value.getType() == pointerType) {
        return value;
    }
    auto cast = mlir::UnrealizedConversionCastOp::create(builder, location,
                                                         pointerType, value);
    return cast.getResult(0);
}

static mlir::Value
createPointerFromMemRef(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value memref, mlir::Value offsetBytes,
                        mlir::NVVM::NVVMMemorySpace memorySpace) {
    if (!memref) {
        return {};
    }
    auto pointerType = getNvvmPointerType(builder, memorySpace);
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(memref.getType())) {
        return castToPointer(builder, location, memref, pointerType);
    }
    if (!isMemRefLike(memref.getType())) {
        return {};
    }

    auto base = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), memref);
    mlir::Value addr = base.getResult();
    if (offsetBytes) {
        auto offset = castToIndex(builder, location, offsetBytes);
        if (!offset) {
            return {};
        }
        addr = mlir::arith::AddIOp::create(builder, location, addr, offset);
    }
    auto addrI64 = mlir::arith::IndexCastOp::create(builder, location,
                                                    builder.getI64Type(), addr);
    return mlir::LLVM::IntToPtrOp::create(builder, location, pointerType,
                                          addrI64.getResult())
        .getResult();
}

static mlir::Value createSharedBarrierPointer(mlir::OpBuilder &builder,
                                              mlir::Location location,
                                              mlir::Value value,
                                              mlir::Value offsetBytes) {
    if (!value) {
        return {};
    }
    auto pointerType =
        getNvvmPointerType(builder, mlir::NVVM::NVVMMemorySpace::Shared);
    if (isMemRefLike(value.getType()) ||
        mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
        return createPointerFromMemRef(builder, location, value, offsetBytes,
                                       mlir::NVVM::NVVMMemorySpace::Shared);
    }
    return castToPointer(builder, location, value, pointerType);
}

static mlir::Value createDescriptorPointer(mlir::OpBuilder &builder,
                                           mlir::Location location,
                                           mlir::Value descriptor) {
    return castToPointer(
        builder, location, descriptor,
        mlir::LLVM::LLVMPointerType::get(builder.getContext()));
}

} // namespace

// NVVM Intrinsics Module
class NVVMIntrinsic : public NamedModule {
  public:
    explicit NVVMIntrinsic();

    void Initialize() override;
    void DeclareModules(mlir::ModuleOp module) override;

    struct MMAConfig {
        int m, n, k;
        MMATypes type_a, type_b;
        MMALayout layout_a, layout_b;
        int fragments_a_count, fragments_b_count, fragments_c_count;
        int fragment_size;
    };

    mlir::Value CreateMma16x8x16F16F16Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMma16x8x16BF16F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMma16x8x8F16F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateSharedAddressFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateOpaqueI32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckOpaqueI32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateLdMatrixKDAFactorsFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckLdMatrixKDAFactorsFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaFenceAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaGroupSyncAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaWaitGroupSyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateSetMaxRegisterFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args,
        mlir::NVVM::SetMaxRegisterAction action,
        llvm::StringRef name) const;

    mlir::Value CreateElectSyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateNamedBarrierFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args,
        bool arrive) const;

    mlir::Value CreateSyncWarpFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateClusterSyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args,
        llvm::StringRef name) const;

    mlir::Value CreateClusterBlockRankFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateGriddepcontrolWaitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateFenceProxyAsyncSharedCTAFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMakeWGMMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args,
        bool returnRawBits = false) const;

    mlir::Value CreateWgmmaInitResultFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaInitResultFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaM64NK16F32BF16BF16Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int64_t n) const;

    mlir::Value CreateWgmmaM64NK16F32BF16BF16RSFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int64_t n) const;

    mlir::Value CreateWgmmaM64N16K16F32BF16BF16RSX8Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaM64N16K16F32BF16BF16RSSmemX8Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaM64N16K16F32BF16BF16RSF32X8Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaM64N16M64N64X2K16F32BF16BF16RSBranchFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaM64NK16F32BF16BF16RSPredicatedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int64_t n) const;

    mlir::Value CreateWgmmaM64N128K128F32BF16BF16Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, bool registerShared,
        int64_t n, int64_t k) const;

    mlir::Value CreateWgmmaM64N192K32F32E4M3E4M3Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateFloatToFP8Function(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args,
                             bool packed) const;

    mlir::Value CreateFloatX2ToBF16X2Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateBF16X2ToFloatX2Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreatePackedBF16MathFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, size_t operandCount,
        llvm::StringRef instruction) const;

    mlir::Value CreateF32MathFunction(ast::Call *call_expr,
                                      GeneratorContext *ctx,
                                      llvm::ArrayRef<mlir::Value> resolved_args,
                                      bool binary,
                                      llvm::StringRef instruction) const;

    mlir::Value
    CreateF32FmaFunction(ast::Call *call_expr, GeneratorContext *ctx,
                         llvm::ArrayRef<mlir::Value> resolved_args,
                         bool fast) const;

    bool CheckWgmmaM64NK16F32BF16BF16RSFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int64_t n,
        llvm::StringRef name) const;

    bool CheckWgmmaM64N16K16F32BF16BF16RSX8Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaM64N16K16F32BF16BF16RSSmemX8Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaM64N16K16F32BF16BF16RSF32X8Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaM64N16M64N64X2K16F32BF16BF16RSBranchFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaM64NK16F32BF16BF16RSPredicatedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int64_t n,
        llvm::StringRef name) const;

    bool CheckWgmmaM64N128K128F32BF16BF16Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, bool registerShared,
        int64_t n, int64_t k) const;

    bool CheckRawWgmmaFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args,
                               int64_t accumulatorSize,
        llvm::StringRef name) const;

    bool CheckFloatToFP8Function(ast::Call *call_expr, GeneratorContext *ctx,
                                 llvm::ArrayRef<mlir::Value> resolved_args,
                                 bool packed) const;

    mlir::Value CreateMakeTMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateTMAFenceFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateTMAPrefetchDescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateTMALoadFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateTMAStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateWgmmaAsyncFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaInitAccumulatorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateWgmmaStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierCreateFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateMBarrierInitFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierTryWaitParityFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierArriveFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierArriveClusterFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierTestWaitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierArriveExpectTxFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateCpAsyncSharedGlobalFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, bool cacheAll) const;

    mlir::Value CreateCpAsyncCommitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateCpAsyncWaitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateGlobalAtomicAddFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateStoreGlobalV4U32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateLoadSharedV4U32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateLoadSharedV2F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateCopySharedToGlobalV4U32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateDrainKDAOutputFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateStoreKDAFactorBlockFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateCpAsyncBulkFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args,
                              CpAsyncBulkIntrinsicKind kind) const;

  private:
    void AddLdMatrixFactory(const std::string &name, const std::string &shape,
                            int num, int bit_width, bool transpose);

    void AddStMatrixFactory(const std::string &name, const std::string &shape,
                            int num, int bit_width, bool transpose);

    mlir::Value
    CreateGenericMMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args,
                             const MMAConfig &config) const;

    mlir::Value
    CreateLdMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args,
                            const std::string &shape, int num, int bit_width,
                            bool transpose = false) const;

    mlir::Value
    CreateStMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args,
                            const std::string &shape, int num, int bit_width,
                            bool transpose = false) const;

    bool
    CheckGenericMMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckLdMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                                llvm::ArrayRef<mlir::Value> resolved_args,
                                const std::string &shape, int num,
                                int bit_width, bool transpose) const;

    bool CheckStMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                                llvm::ArrayRef<mlir::Value> resolved_args,
                                const std::string &shape, int num,
                                int bit_width, bool transpose) const;
    bool CheckWgmmaFenceAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaGroupSyncAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaWaitGroupSyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckSetMaxRegisterFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args,
        llvm::StringRef name) const;

    bool CheckElectSyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckNamedBarrierFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args,
        llvm::StringRef name) const;

    bool CheckSyncWarpFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckNoArgumentFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args,
        llvm::StringRef name) const;

    bool CheckFenceProxyAsyncSharedCTAFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMakeWGMMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool
    CheckWgmmaAsyncFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaInitAccumulatorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool
    CheckWgmmaStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierCreateFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool
    CheckMBarrierInitFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierTryWaitParityFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierArriveFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierArriveClusterFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierTestWaitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierArriveExpectTxFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCpAsyncCaSharedGlobalFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCpAsyncCommitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCpAsyncWaitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckGlobalAtomicAddFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckStoreGlobalV4U32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckLoadSharedV4U32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckLoadSharedV2F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCopySharedToGlobalV4U32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckDrainKDAOutputFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckStoreKDAFactorBlockFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCpAsyncBulkFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                  llvm::ArrayRef<mlir::Value> resolved_args,
                                  CpAsyncBulkIntrinsicKind kind) const;

    bool CheckMakeTMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckTMAFenceFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckTMAPrefetchDescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckTMALoadFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckTMAStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;
};

NVVMIntrinsic::NVVMIntrinsic() : NamedModule("nvvm") {}

void NVVMIntrinsic::Initialize() {
    // Add MMA sync intrinsic function helpers with constraints encoded in
    // function names
    AddFunction(
        "mma_16x8x16_f16_f16",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMma16x8x16F16F16Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericMMAFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "mma_16x8x16_bf16_f32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMma16x8x16BF16F32Function(call_expr, gen_ctx,
                                                   resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericMMAFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "mma_16x8x8_f16_f32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMma16x8x8F16F32Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericMMAFunction(call_expr, gen_ctx, resolved_args);
        });

    // LLVM 22 models ldmatrix/stmatrix shape and element type explicitly.
    // The implemented wrappers cover the standard m8n8/b16 family; the old
    // b8 and m16n16 registrations do not match the stricter NVVM verifier.
    for (int num : {1, 2, 4}) {
        std::string base_name =
            "ldmatrix_m8n8_x" + std::to_string(num) + "_b16";
        AddLdMatrixFactory(base_name, "m8n8", num, 16, false);
        AddLdMatrixFactory(base_name + "_trans", "m8n8", num, 16, true);
    }

    AddFunction(
        "ldmatrix_m8n8_x4_b16_kda_factors",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateLdMatrixKDAFactorsFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckLdMatrixKDAFactorsFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "shared_address",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSharedAddressFunction(call_expr, gen_ctx,
                                               resolved_args);
        },
        [](ast::Call *, GeneratorContext *,
           llvm::ArrayRef<mlir::Value>) -> bool { return true; });

    AddFunction(
        "opaque_i32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateOpaqueI32Function(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckOpaqueI32Function(call_expr, gen_ctx, resolved_args);
        });

    for (int num : {1, 2, 4}) {
        std::string base_name =
            "stmatrix_m8n8_x" + std::to_string(num) + "_b16";
        AddStMatrixFactory(base_name, "m8n8", num, 16, false);
        AddStMatrixFactory(base_name + "_trans", "m8n8", num, 16, true);
    }

    AddFunction(
        "wgmma_fence_aligned",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaFenceAlignedFunction(call_expr, gen_ctx,
                                                   resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaFenceAlignedFunction(call_expr, gen_ctx,
                                                  resolved_args);
        });

    AddFunction(
        "wgmma_group_sync_aligned",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaGroupSyncAlignedFunction(call_expr, gen_ctx,
                                                       resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaGroupSyncAlignedFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "wgmma_wait_group_sync",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaWaitGroupSyncFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaWaitGroupSyncFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "setmaxnreg_inc",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSetMaxRegisterFunction(
                call_expr, gen_ctx, resolved_args,
                mlir::NVVM::SetMaxRegisterAction::increase,
                "setmaxnreg_inc");
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSetMaxRegisterFunction(call_expr, gen_ctx,
                                               resolved_args,
                                               "setmaxnreg_inc");
        });

    AddFunction(
        "setmaxnreg_dec",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSetMaxRegisterFunction(
                call_expr, gen_ctx, resolved_args,
                mlir::NVVM::SetMaxRegisterAction::decrease,
                "setmaxnreg_dec");
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSetMaxRegisterFunction(call_expr, gen_ctx,
                                               resolved_args,
                                               "setmaxnreg_dec");
        });

    AddFunction(
        "elect_sync",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateElectSyncFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckElectSyncFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "named_barrier_sync",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateNamedBarrierFunction(call_expr, gen_ctx,
                                              resolved_args,
                                              /*arrive=*/false);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckNamedBarrierFunction(call_expr, gen_ctx,
                                             resolved_args,
                                             "named_barrier_sync");
        });

    AddFunction(
        "named_barrier_arrive",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateNamedBarrierFunction(call_expr, gen_ctx,
                                              resolved_args,
                                              /*arrive=*/true);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckNamedBarrierFunction(call_expr, gen_ctx,
                                             resolved_args,
                                             "named_barrier_arrive");
        });

    AddFunction(
        "syncwarp",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSyncWarpFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSyncWarpFunction(call_expr, gen_ctx, resolved_args);
        });

    for (llvm::StringRef name : {"fence_mbarrier_init_release_cluster",
                                 "cluster_arrive_relaxed", "cluster_wait"}) {
        AddFunction(
            name.str(),
            [this, name](ast::Call *call_expr, GeneratorContext *gen_ctx,
                         llvm::ArrayRef<mlir::Value> resolved_args)
                -> mlir::Value {
                return CreateClusterSyncFunction(call_expr, gen_ctx,
                                                 resolved_args, name);
            },
            [this, name](ast::Call *call_expr, GeneratorContext *gen_ctx,
                         llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckNoArgumentFunction(call_expr, gen_ctx,
                                               resolved_args, name);
            });
    }

    AddFunction(
        "cluster_block_rank",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateClusterBlockRankFunction(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckNoArgumentFunction(call_expr, gen_ctx, resolved_args,
                                           "cluster_block_rank");
        });

    AddFunction(
        "griddepcontrol_wait",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateGriddepcontrolWaitFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckNoArgumentFunction(call_expr, gen_ctx, resolved_args,
                                           "griddepcontrol_wait");
        });

    AddFunction(
        "fence_proxy_async_shared_cta",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateFenceProxyAsyncSharedCTAFunction(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckFenceProxyAsyncSharedCTAFunction(
                call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "make_wgmma_descriptor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeWGMMADescriptorFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeWGMMADescriptorFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "make_wgmma_descriptor_bits",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeWGMMADescriptorFunction(
                call_expr, gen_ctx, resolved_args, /*returnRawBits=*/true);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeWGMMADescriptorFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "wgmma_init_result",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaInitResultFunction(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaInitResultFunction(call_expr, gen_ctx,
                                                resolved_args);
        });

    AddFunction(
        "wgmma_m64n64k16_f32_bf16_bf16",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaM64NK16F32BF16BF16Function(call_expr, gen_ctx,
                                                         resolved_args, 64);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckRawWgmmaFunction(call_expr, gen_ctx, resolved_args, 32,
                "wgmma_m64n64k16_f32_bf16_bf16");
        });

    AddFunction(
        "wgmma_m64n128k16_f32_bf16_bf16",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaM64NK16F32BF16BF16Function(call_expr, gen_ctx,
                                                         resolved_args, 128);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckRawWgmmaFunction(call_expr, gen_ctx, resolved_args, 64,
                                         "wgmma_m64n128k16_f32_bf16_bf16");
        });

    for (int64_t n : {96, 160, 176, 192}) {
        std::string name =
            "wgmma_m64n" + std::to_string(n) + "k16_f32_bf16_bf16";
        AddFunction(
            name,
            [this, n](ast::Call *call_expr, GeneratorContext *gen_ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateWgmmaM64NK16F32BF16BF16Function(
                    call_expr, gen_ctx, resolved_args, n);
            },
            [this, n, name](
                ast::Call *call_expr, GeneratorContext *gen_ctx,
                llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckRawWgmmaFunction(call_expr, gen_ctx, resolved_args,
                                             n / 2, name);
            });
    }

    for (int64_t n : {16, 64, 128}) {
        std::string name =
            "wgmma_m64n" + std::to_string(n) + "k16_f32_bf16_bf16_rs";
        AddFunction(
            name,
            [this, n](ast::Call *call_expr, GeneratorContext *gen_ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateWgmmaM64NK16F32BF16BF16RSFunction(
                    call_expr, gen_ctx, resolved_args, n);
            },
            [this, n, name](
                ast::Call *call_expr, GeneratorContext *gen_ctx,
                llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckWgmmaM64NK16F32BF16BF16RSFunction(
                    call_expr, gen_ctx, resolved_args, n, name);
            });
    }

    AddFunction(
        "wgmma_m64n16k16_f32_bf16_bf16_rs_smem_x8",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaM64N16K16F32BF16BF16RSSmemX8Function(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaM64N16K16F32BF16BF16RSSmemX8Function(
                call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "wgmma_m64n16x2_m64n64x4_k16_f32_bf16_bf16_rs_branch",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaM64N16M64N64X2K16F32BF16BF16RSBranchFunction(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaM64N16M64N64X2K16F32BF16BF16RSBranchFunction(
                call_expr, gen_ctx, resolved_args);
        });

    for (auto [name, registerShared, n, k] :
         {std::tuple{"wgmma_m64n128k128_f32_bf16_bf16_ss", false, 128, 128},
          std::tuple{"wgmma_m64n176k128_f32_bf16_bf16_ss", false, 176, 128},
          std::tuple{"wgmma_m64n192k128_f32_bf16_bf16_ss", false, 192, 128},
          std::tuple{"wgmma_m64n128k128_f32_bf16_bf16_rs", true, 128, 128},
          std::tuple{"wgmma_m64n128k176_f32_bf16_bf16_rs", true, 128, 176},
          std::tuple{"wgmma_m64n128k192_f32_bf16_bf16_rs", true, 128, 192}}) {
        AddFunction(
            name,
            [this, registerShared, n,
             k](ast::Call *call_expr, GeneratorContext *gen_ctx,
                llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateWgmmaM64N128K128F32BF16BF16Function(
                    call_expr, gen_ctx, resolved_args, registerShared, n, k);
            },
            [this, registerShared, n,
             k](ast::Call *call_expr, GeneratorContext *gen_ctx,
                llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckWgmmaM64N128K128F32BF16BF16Function(
                    call_expr, gen_ctx, resolved_args, registerShared, n, k);
            });
    }

    AddFunction(
        "wgmma_m64n192k32_f32_e4m3_e4m3",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaM64N192K32F32E4M3E4M3Function(call_expr, gen_ctx,
                                                            resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckRawWgmmaFunction(call_expr, gen_ctx, resolved_args, 96,
                "wgmma_m64n192k32_f32_e4m3_e4m3");
        });

    for (auto [name, packed] : {std::pair{"float_to_fp8", false},
          std::pair{"floatx2_to_fp8x2", true}}) {
        AddFunction(
            name,
            [this,
             packed](ast::Call *call_expr, GeneratorContext *gen_ctx,
                     llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateFloatToFP8Function(call_expr, gen_ctx,
                                                resolved_args, packed);
            },
            [this, packed](ast::Call *call_expr, GeneratorContext *gen_ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckFloatToFP8Function(call_expr, gen_ctx,
                                               resolved_args, packed);
            });
    }

    AddFunction(
        "floatx2_to_bf16x2",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateFloatX2ToBF16X2Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [](ast::Call *, GeneratorContext *,
           llvm::ArrayRef<mlir::Value>) -> bool { return true; });

    AddFunction(
        "bf16x2_to_floatx2",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBF16X2ToFloatX2Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [](ast::Call *, GeneratorContext *,
           llvm::ArrayRef<mlir::Value>) -> bool { return true; });

    for (auto [name, operandCount, instruction] :
         {std::tuple{"bf16x2_add", size_t{2},
                     "add.rn.bf16x2 $0, $1, $2;"},
          std::tuple{"bf16x2_sub", size_t{2},
                     "sub.rn.bf16x2 $0, $1, $2;"},
          std::tuple{"bf16x2_mul", size_t{2},
                     "mul.rn.bf16x2 $0, $1, $2;"},
          std::tuple{"bf16x2_fma", size_t{3},
                     "fma.rn.bf16x2 $0, $1, $2, $3;"}}) {
        AddFunction(
            name,
            [this, operandCount, instruction](
                ast::Call *call_expr, GeneratorContext *gen_ctx,
                llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreatePackedBF16MathFunction(
                    call_expr, gen_ctx, resolved_args, operandCount,
                    instruction);
            },
            [](ast::Call *, GeneratorContext *,
               llvm::ArrayRef<mlir::Value>) -> bool { return true; });
    }

    for (auto [name, binary, instruction] :
         {std::tuple{"fmax", true, "max.f32 $0, $1, $2;"},
          std::tuple{"fast_fmax", true, "max.ftz.f32 $0, $1, $2;"},
          std::tuple{"fast_exp2", false, "ex2.approx.ftz.f32 $0, $1;"},
          std::tuple{"fast_log2", false, "lg2.approx.ftz.f32 $0, $1;"},
          std::tuple{"fast_tanh", false, "tanh.approx.f32 $0, $1;"},
          std::tuple{"fast_rsqrt", false, "rsqrt.approx.ftz.f32 $0, $1;"},
          std::tuple{"fast_rcp", false, "rcp.approx.ftz.f32 $0, $1;"}}) {
        AddFunction(
            name,
            [this,
             binary,
             instruction](ast::Call *call_expr, GeneratorContext *gen_ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args)
                -> mlir::Value {
                return CreateF32MathFunction(call_expr, gen_ctx, resolved_args,
                                             binary, instruction);
            },
            [](ast::Call *, GeneratorContext *,
               llvm::ArrayRef<mlir::Value>) -> bool { return true; });
    }

    for (auto [name, fast] :
         {std::pair{"fma", false}, std::pair{"fast_fma", true}}) {
        AddFunction(
            name,
            [this,
             fast](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateF32FmaFunction(call_expr, gen_ctx, resolved_args,
                                            fast);
            },
            [](ast::Call *, GeneratorContext *,
               llvm::ArrayRef<mlir::Value>) -> bool { return true; });
    }

    AddFunction(
        "wgmma_async",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaAsyncFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaAsyncFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "wgmma_init_accumulator",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaInitAccumulatorFunction(call_expr, gen_ctx,
                                                      resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaInitAccumulatorFunction(call_expr, gen_ctx,
                                                     resolved_args);
        });

    AddFunction(
        "wgmma_store",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaStoreFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaStoreFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "mbarrier_create",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierCreateFunction(call_expr, gen_ctx,
                                                resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierCreateFunction(call_expr, gen_ctx,
                                               resolved_args);
        });

    AddFunction(
        "mbarrier_init",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierInitFunction(call_expr, gen_ctx,
                                              resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierInitFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "mbarrier_try_wait_parity",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierTryWaitParityFunction(call_expr, gen_ctx,
                                                       resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierTryWaitParityFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "mbarrier_arrive",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierArriveFunction(call_expr, gen_ctx,
                                                resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierArriveFunction(call_expr, gen_ctx,
                                               resolved_args);
        });

    AddFunction(
        "mbarrier_arrive_cluster",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierArriveClusterFunction(call_expr, gen_ctx,
                                                       resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierArriveClusterFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "mbarrier_test_wait",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierTestWaitFunction(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierTestWaitFunction(call_expr, gen_ctx,
                                                 resolved_args);
        });

    AddFunction(
        "mbarrier_arrive_expect_tx",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierArriveExpectTxFunction(call_expr, gen_ctx,
                                                        resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierArriveExpectTxFunction(call_expr, gen_ctx,
                                                       resolved_args);
        });

    for (auto [name, cacheAll] :
         {std::pair{"cp_async_ca_shared_global", true},
          std::pair{"cp_async_cg_shared_global", false}}) {
        AddFunction(
            name,
            [this, cacheAll](ast::Call *call_expr,
                             GeneratorContext *gen_ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args)
                -> mlir::Value {
                return CreateCpAsyncSharedGlobalFunction(
                    call_expr, gen_ctx, resolved_args, cacheAll);
            },
            [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckCpAsyncCaSharedGlobalFunction(
                    call_expr, gen_ctx, resolved_args);
            });
    }

    AddFunction(
        "cp_async_commit_group",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateCpAsyncCommitGroupFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckCpAsyncCommitGroupFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "cp_async_wait_group",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateCpAsyncWaitGroupFunction(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckCpAsyncWaitGroupFunction(call_expr, gen_ctx,
                                                 resolved_args);
        });

    auto addCpAsyncBulkFunction = [this](llvm::StringRef name,
                                         CpAsyncBulkIntrinsicKind kind) {
        AddFunction(
            name.str(),
            [this,
             kind](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateCpAsyncBulkFunction(call_expr, gen_ctx,
                                                 resolved_args, kind);
            },
            [this, kind](ast::Call *call_expr, GeneratorContext *gen_ctx,
                         llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckCpAsyncBulkFunction(call_expr, gen_ctx,
                                                resolved_args, kind);
            });
    };

    addCpAsyncBulkFunction("cp_async_bulk_commit_group",
                           CpAsyncBulkIntrinsicKind::CommitGroup);
    addCpAsyncBulkFunction("cp_async_bulk_global_shared_cta",
                           CpAsyncBulkIntrinsicKind::GlobalSharedCTA);
    addCpAsyncBulkFunction("cp_async_bulk_prefetch",
                           CpAsyncBulkIntrinsicKind::Prefetch);
    addCpAsyncBulkFunction("cp_async_bulk_shared_cluster_global",
                           CpAsyncBulkIntrinsicKind::SharedClusterGlobal);
    addCpAsyncBulkFunction("cp_async_bulk_shared_cluster_shared_cta",
                           CpAsyncBulkIntrinsicKind::SharedClusterSharedCTA);
    addCpAsyncBulkFunction("cp_async_bulk_tensor_global_shared_cta",
                           CpAsyncBulkIntrinsicKind::TensorGlobalSharedCTA);
    addCpAsyncBulkFunction("cp_async_bulk_tensor_prefetch",
                           CpAsyncBulkIntrinsicKind::TensorPrefetch);
    addCpAsyncBulkFunction("cp_async_bulk_tensor_reduce",
                           CpAsyncBulkIntrinsicKind::TensorReduce);
    addCpAsyncBulkFunction("cp_async_bulk_tensor_shared_cluster_global",
                           CpAsyncBulkIntrinsicKind::TensorSharedClusterGlobal);
    addCpAsyncBulkFunction("cp_async_bulk_wait_group",
                           CpAsyncBulkIntrinsicKind::WaitGroup);

    AddFunction(
        "make_tma_descriptor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeTMADescriptorFunction(call_expr, gen_ctx,
                                                   resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeTMADescriptorFunction(call_expr, gen_ctx,
                                                  resolved_args);
        });

    AddFunction(
        "tma_fence",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMAFenceFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMAFenceFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "tma_prefetch_descriptor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMAPrefetchDescriptorFunction(call_expr, gen_ctx,
                                                       resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMAPrefetchDescriptorFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "tma_load",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMALoadFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMALoadFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "tma_store",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMAStoreFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMAStoreFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "store_global_v4_u32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateStoreGlobalV4U32Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckStoreGlobalV4U32Function(call_expr, gen_ctx,
                                                 resolved_args);
        });

    AddFunction(
        "load_shared_v2_f32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateLoadSharedV2F32Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckLoadSharedV2F32Function(call_expr, gen_ctx,
                                                resolved_args);
        });

    AddFunction(
        "drain_kda_output",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateDrainKDAOutputFunction(call_expr, gen_ctx,
                                                resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckDrainKDAOutputFunction(call_expr, gen_ctx,
                                               resolved_args);
        });

    AddFunction(
        "atomic_add",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateGlobalAtomicAddFunction(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGlobalAtomicAddFunction(call_expr, gen_ctx,
                                                resolved_args);
        });
}

void NVVMIntrinsic::DeclareModules(mlir::ModuleOp module) {
    if (!module)
        return;

    // Register the bytecode in the intrinsic registry
    auto libraryBytes = GetNvvmIntrinsicLibrary();
    auto &registry = utils::EmbeddedFilesystemView::getInstance();
    registry.registerFile(std::string(kNvvmIntrinsicLibraryName), libraryBytes);

    intrinsics::GetOrCreateImplementationContainer(module, "nvvm",
                                                   kNvvmIntrinsicLibraryTag);

    auto loadDialects = [](mlir::MLIRContext *ctx) {
        ctx->loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
                         mlir::memref::MemRefDialect,
                         mlir::vector::VectorDialect, cf::AveLangDialect>();
    };

    if (failed(intrinsics::EnsureIntrinsicDeclarations(
            module, kNvvmIntrinsicLibraryName, libraryBytes, loadDialects))) {
        module.emitError() << "failed to declare NVVM intrinsics";
    }
}

void NVVMIntrinsic::AddLdMatrixFactory(const std::string &name,
                                       const std::string &shape, int num,
                                       int bit_width, bool transpose) {
    AddFunction(
        name,
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateLdMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                           shape, num, bit_width, transpose);
        },
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckLdMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                          shape, num, bit_width, transpose);
        });
}

void NVVMIntrinsic::AddStMatrixFactory(const std::string &name,
                                       const std::string &shape, int num,
                                       int bit_width, bool transpose) {
    AddFunction(
        name,
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateStMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                           shape, num, bit_width, transpose);
        },
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckStMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                          shape, num, bit_width, transpose);
        });
}

mlir::Value NVVMIntrinsic::CreateGenericMMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const MMAConfig &config) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto a = resolved_args[0];
    auto b = resolved_args[1];
    auto c = resolved_args[2];

    auto c_vector = mlir::dyn_cast<mlir::VectorType>(c.getType());
    if (!c_vector) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "C operand for MMA operation must be a vector type";
        return nullptr;
    }

    // Determine result type based on configuration
    mlir::Type result_type;
    if (c_vector.getElementType().isF16()) {
        result_type = c_vector;
    } else if (c_vector.getElementType().isF32()) {
        result_type = c_vector;
    } else {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Unsupported result type for MMA operation";
        return nullptr;
    }

    // Create GPUOp NVVM MMA operation
    auto mma_op =
        cf::NVVMMMAOp::create(builder, location, result_type, a, b, c);

    return mma_op.getResult();
}

mlir::Value NVVMIntrinsic::CreateMma16x8x16F16F16Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    MMAConfig config = {.m = 16,
                        .n = 8,
                        .k = 16,
                        .type_a = MMATypes::f16,
                        .type_b = MMATypes::f16,
                        .layout_a = MMALayout::row,
                        .layout_b = MMALayout::col,
                        .fragments_a_count = 4,
                        .fragments_b_count = 2,
                        .fragments_c_count = 2,
                        .fragment_size = 2};
    return CreateGenericMMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value NVVMIntrinsic::CreateMma16x8x16BF16F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    MMAConfig config = {.m = 16,
                        .n = 8,
                        .k = 16,
                        .type_a = MMATypes::f16,
                        .type_b = MMATypes::f32,
                        .layout_a = MMALayout::row,
                        .layout_b = MMALayout::col,
                        .fragments_a_count = 4,
                        .fragments_b_count = 2,
                        .fragments_c_count = 4,
                        .fragment_size = 2};
    return CreateGenericMMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value NVVMIntrinsic::CreateMma16x8x8F16F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    MMAConfig config = {.m = 16,
                        .n = 8,
                        .k = 8,
                        .type_a = MMATypes::f16,
                        .type_b = MMATypes::f32,
                        .layout_a = MMALayout::row,
                        .layout_b = MMALayout::col,
                        .fragments_a_count = 2,
                        .fragments_b_count = 1,
                        .fragments_c_count = 4,
                        .fragment_size = 2};
    return CreateGenericMMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value NVVMIntrinsic::CreateLdMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto memref_ptr = resolved_args[0];

    if (memref_ptr.getType().isInteger(32)) {
        mlir::Type result_type = builder.getI32Type();
        if (num > 1) {
            llvm::SmallVector<mlir::Type> resultElements(
                num, builder.getI32Type());
            result_type = mlir::LLVM::LLVMStructType::getLiteral(
                builder.getContext(), resultElements);
        }
        std::string outputs = "{";
        std::string constraints;
        for (int index = 0; index < num; ++index) {
            if (index != 0) {
                outputs += ", ";
                constraints += ",";
            }
            outputs += "$" + std::to_string(index);
            constraints += "=r";
        }
        outputs += "}";
        constraints += ",r";
        std::string instruction =
            "ldmatrix.sync.aligned.m8n8.x" + std::to_string(num) +
            (transpose ? ".trans.shared.b16 " : ".shared.b16 ") + outputs +
            ", [$" + std::to_string(num) + "];";
        auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
            builder, location, result_type, resolved_args, instruction,
            constraints, /*hasSideEffects=*/true, /*isAlignStack=*/false,
            mlir::LLVM::tailcallkind::TailCallKind::None,
            mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
        if (num == 1) {
            return inlineAsm.getRes();
        }
        llvm::SmallVector<mlir::Value> results;
        for (int index = 0; index < num; ++index) {
            results.push_back(mlir::LLVM::ExtractValueOp::create(
                builder, location, builder.getI32Type(), inlineAsm.getRes(),
                llvm::ArrayRef<int64_t>{index}));
        }
        return mlir::vector::FromElementsOp::create(
            builder, location,
            mlir::VectorType::get({num}, builder.getI32Type()), results);
    }

    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref_ptr.getType());
    if (!memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix expects a memref argument";
        return nullptr;
    }

    auto is_compatible_memref = [&](const std::string &shape_tag) {
        auto gpu_space = mlir::gpu::AddressSpaceAttr::get(
            builder.getContext(), mlir::gpu::AddressSpace::Workgroup);

        if (memref_type.getMemorySpace() != gpu_space) {
            return false;
        }

        auto elem_type = memref_type.getElementType();

        if (bit_width == 16) {
            bool valid_type = elem_type.isF16() || elem_type.isBF16() ||
                              elem_type.isInteger(16);
            if (!valid_type) {
                return false;
            }
        } else if (bit_width == 8) {
            bool valid_type = elem_type.isInteger(8) ||
                              isa<Float8E4M3FNType>(elem_type) ||
                              isa<Float8E4M3FNUZType>(elem_type);
            if (!valid_type) {
                return false;
            }
        }

        auto shape_vec = memref_type.getShape();
        if (shape_tag == "m8n8") {
            return shape_vec.size() == 2 && shape_vec[0] == 8 &&
                   shape_vec[1] == 8;
        }
        if (shape_tag == "m16n16") {
            return shape_vec.size() == 2 && shape_vec[0] == 16 &&
                   shape_vec[1] == 16;
        }
        return false;
    };

    auto type_to_string = [](mlir::Type type) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        type.print(os);
        return buffer;
    };

    if (!is_compatible_memref(shape)) {
        std::string expected_shape = shape == "m8n8" ? "8x8" : "16x16";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape << " expects memref<" << expected_shape
            << "xb" << bit_width
            << ", #gpu.address_space<workgroup>> (with optional "
               "strides/offset) but found "
            << type_to_string(memref_ptr.getType());
        return nullptr;
    }

    mlir::Type result_type;
    if (num == 1) {
        result_type = builder.getI32Type();
    } else {
        result_type = mlir::VectorType::get({num}, builder.getI32Type());
    }

    auto ld_matrix_op = cf::NVVMLdMatrixOp::create(
        builder, location, result_type, memref_ptr,
        mlir::StringAttr::get(builder.getContext(), shape),
        mlir::IntegerAttr::get(builder.getI32Type(), num),
        mlir::IntegerAttr::get(builder.getI32Type(), bit_width),
        builder.getBoolAttr(transpose));

    return ld_matrix_op.getResult();
}

bool NVVMIntrinsic::CheckLdMatrixKDAFactorsFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    bool valid = resolved_args.size() == 5 &&
                 llvm::all_of(resolved_args, [](mlir::Value value) {
                     return value && value.getType().isInteger(32);
                 });
    if (!valid) {
        ctx->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            call_expr->GetSourceRange().getBegin())
            << "ldmatrix_m8n8_x4_b16_kda_factors requires five i32 arguments";
    }
    return valid;
}

mlir::Value NVVMIntrinsic::CreateLdMatrixKDAFactorsFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckLdMatrixKDAFactorsFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }
    constexpr int64_t kOutputs = 20;
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Type> elementTypes(kOutputs,
                                               builder.getI32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    auto outputList = [](int64_t begin) {
        std::string result = "{";
        for (int64_t index = 0; index < 4; ++index) {
            if (index != 0)
                result += ", ";
            result += "$" + std::to_string(begin + index);
        }
        return result + "}";
    };
    constexpr int64_t kBase = kOutputs;
    constexpr int64_t kStage = kBase + 1;
    constexpr int64_t kRow = kStage + 1;
    constexpr int64_t kCol = kRow + 1;
    constexpr int64_t kWarp = kCol + 1;
    std::string asmString =
        "{ .reg .b32 c, rowa, waddr, vaddr, tmp;\n"
        "shl.b32 c, $" + std::to_string(kCol) + ", 1;\n"
        "mad.lo.u32 tmp, $" + std::to_string(kRow) + ", 48, c;\n"
        "add.u32 tmp, tmp, $" + std::to_string(kBase) + ";\n"
        "add.u32 tmp, tmp, 93056;\n"
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 " + outputList(0) +
        ", [tmp];\n"
        "mad.lo.u32 rowa, $" + std::to_string(kRow) + ", 272, c;\n"
        "shl.b32 tmp, $" + std::to_string(kWarp) + ", 6;\n"
        "add.u32 rowa, rowa, tmp;\n"
        "add.u32 waddr, rowa, $" + std::to_string(kBase) + ";\n"
        "add.u32 waddr, waddr, 77440;\n"
        "mad.lo.u32 vaddr, $" + std::to_string(kStage) +
        ", 4352, rowa;\n"
        "add.u32 vaddr, vaddr, $" + std::to_string(kBase) + ";\n"
        "add.u32 vaddr, vaddr, 68608;\n"
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " + outputList(4) +
        ", [waddr];\n"
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " + outputList(8) +
        ", [vaddr];\n"
        "add.u32 waddr, waddr, 32;\n"
        "add.u32 vaddr, vaddr, 32;\n"
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " + outputList(12) +
        ", [waddr];\n"
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 " + outputList(16) +
        ", [vaddr];\n}";
    llvm::SmallVector<std::string> constraints(kOutputs, "=r");
    constraints.insert(constraints.end(), 5, "r");
    constraints.push_back("~{memory}");
    std::string constraintString;
    for (size_t index = 0; index < constraints.size(); ++index) {
        if (index != 0)
            constraintString += ",";
        constraintString += constraints[index];
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, resolved_args, asmString,
        constraintString, /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    llvm::SmallVector<mlir::Value> elements;
    elements.reserve(kOutputs);
    for (int64_t index = 0; index < kOutputs; ++index) {
        elements.push_back(mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getI32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{index}));
    }
    return mlir::vector::FromElementsOp::create(
        builder, location,
        mlir::VectorType::get({kOutputs}, builder.getI32Type()), elements);
}

mlir::Value NVVMIntrinsic::CreateSharedAddressFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1 || !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "shared_address requires one shared memref";
        return nullptr;
    }
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto gpuSpace = mlir::gpu::AddressSpaceAttr::get(
        builder.getContext(), mlir::gpu::AddressSpace::Workgroup);
    if (!memrefType || memrefType.getMemorySpace() != gpuSpace) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "shared_address requires a workgroup memref";
        return nullptr;
    }
    auto location = builder.getUnknownLoc();
    auto address = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), resolved_args[0]);
    return mlir::arith::IndexCastOp::create(
        builder, location, builder.getI32Type(), address.getResult());
}

bool NVVMIntrinsic::CheckOpaqueI32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    bool valid = resolved_args.size() == 1 && resolved_args[0] &&
                 resolved_args[0].getType().isInteger(32);
    if (!valid) {
        ctx->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            call_expr->GetSourceRange().getBegin())
            << "opaque_i32 requires one i32 argument";
    }
    return valid;
}

mlir::Value NVVMIntrinsic::CreateOpaqueI32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckOpaqueI32Function(call_expr, ctx, resolved_args)) {
        return nullptr;
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, builder.getI32Type(),
        mlir::ValueRange{resolved_args[0]}, "mov.u32 $0, $1;",
        "=r,r,~{memory}", /*hasSideEffects=*/true,
        /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return inlineAsm.getRes();
}

mlir::Value NVVMIntrinsic::CreateStMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto memref_ptr = resolved_args[0];

    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref_ptr.getType());
    if (!memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix expects a memref argument";
        return nullptr;
    }

    auto is_compatible_memref = [&](const std::string &shape_tag) {
        auto gpu_space = mlir::gpu::AddressSpaceAttr::get(
            builder.getContext(), mlir::gpu::AddressSpace::Workgroup);

        if (memref_type.getMemorySpace() != gpu_space) {
            return false;
        }

        auto elem_type = memref_type.getElementType();

        if (bit_width == 16) {
            bool valid_type = elem_type.isF16() || elem_type.isBF16() ||
                              elem_type.isInteger(16);
            if (!valid_type) {
                return false;
            }
        } else if (bit_width == 8) {
            bool valid_type = elem_type.isInteger(8) ||
                              isa<Float8E4M3FNType>(elem_type) ||
                              isa<Float8E4M3FNUZType>(elem_type);
            if (!valid_type) {
                return false;
            }
        }

        // PTX stmatrix consumes only the shared-memory address named by each
        // lane. High-performance swizzled layouts therefore pass a subview at
        // the lane-specific address rather than a logical 8x8 memref.
        return !memref_type.getShape().empty();
    };

    auto type_to_string = [](mlir::Type type) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        type.print(os);
        return buffer;
    };

    if (!is_compatible_memref(shape)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << " expects a non-empty shared memref "
            << "with b" << bit_width << " elements but found "
            << type_to_string(memref_ptr.getType());
        return nullptr;
    }

    auto source_arg = resolved_args[1];
    if (!source_arg) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate source operand for stmatrix_" << shape;
        return nullptr;
    }

    auto source_type = source_arg.getType();

    if (num == 1) {
        if (!source_type.isInteger(32) && source_type.isInteger()) {
            source_arg = mlir::arith::ExtSIOp::create(
                builder, location, builder.getI32Type(), source_arg);
            source_type = source_arg.getType();
        }
    } else {
        auto vector_type = mlir::cast<mlir::VectorType>(source_type);
        (void)vector_type;
    }

    cf::NVVMStMatrixOp::create(
        builder, location, source_arg, memref_ptr,
        mlir::StringAttr::get(builder.getContext(), shape),
        mlir::IntegerAttr::get(builder.getI32Type(), num),
        mlir::IntegerAttr::get(builder.getI32Type(), bit_width),
        builder.getBoolAttr(transpose));

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckGenericMMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mma operation requires exactly 3 arguments: a, b, c";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mma operation";
        return false;
    }

    if (!mlir::dyn_cast<mlir::VectorType>(resolved_args[0].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[1].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[2].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mma operands must be vector types";
        return false;
    }

    return true;
}

bool NVVMIntrinsic::CheckLdMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape << "_x" << num << "_b" << bit_width
            << (transpose ? "_trans" : "")
            << " requires exactly 1 argument: ptr";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate pointer operand for ldmatrix_" << shape;
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType()) &&
        !resolved_args[0].getType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape
            << " expects a shared memref or i32 shared address";
        return false;
    }

    return true;
}

bool NVVMIntrinsic::CheckStMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    size_t expected_args = 2;
    if (resolved_args.size() != expected_args) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << "_x" << num << "_b" << bit_width
            << (transpose ? "_trans" : "") << " requires exactly "
            << expected_args << " arguments: ptr + vector source values";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for stmatrix_" << shape;
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << " expects memref pointer operand";
        return false;
    }

    auto source_type = resolved_args[1].getType();
    if (num == 1) {
        if (!source_type.isInteger(32)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "stmatrix_" << shape
                << "_x1 source operand must be integer type";
            return false;
        }
    } else {
        auto vector_type = mlir::dyn_cast<mlir::VectorType>(source_type);
        if (!vector_type || vector_type.getNumElements() != num ||
            !vector_type.getElementType().isInteger(32)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "stmatrix_" << shape << "_x" << num
                << " source operand must be i32x" << num << " vector type";
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaFenceAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    emitInlinePtxVoid(builder, location, "wgmma.fence.sync.aligned;",
                      "~{memory}");
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaFenceAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_fence_aligned requires no arguments";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaGroupSyncAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    emitInlinePtxVoid(builder, location, "wgmma.commit_group.sync.aligned;",
                      "~{memory}");
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaGroupSyncAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_group_sync_aligned requires no arguments";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaWaitGroupSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaWaitGroupSyncFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto groupValue = ConstantFolder::FoldIntValue(resolved_args[0]);
    if (!groupValue) {
        return nullptr;
    }

    std::string asmString =
        "wgmma.wait_group.sync.aligned " + std::to_string(*groupValue) + ";";
    emitInlinePtxVoid(builder, location, asmString, "~{memory}");
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaWaitGroupSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync requires exactly 1 argument: group";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate group operand for wgmma_wait_group_sync";
        return false;
    }

    if (!resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync group operand must be integer type";
        return false;
    }

    if (!ConstantFolder::FoldIntValue(resolved_args[0])) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync requires a constant integer value for "
               "group";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateSetMaxRegisterFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    mlir::NVVM::SetMaxRegisterAction action, llvm::StringRef name) const {
    if (!CheckSetMaxRegisterFunction(call_expr, ctx, resolved_args, name)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    mlir::NVVM::SetMaxRegisterOp::create(
        builder, location,
        static_cast<uint32_t>(*getConstantIntValue(resolved_args[0])), action);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckSetMaxRegisterFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, llvm::StringRef name) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << name << " " << message;
        return false;
    };
    if (resolved_args.size() != 1 || !resolved_args[0] ||
        !resolved_args[0].getType().isIntOrIndex()) {
        return report("requires one constant integer register count");
    }
    auto count = getConstantIntValue(resolved_args[0]);
    if (!count) {
        return report("register count must be a constant integer");
    }
    if (*count < 24 || *count > 256 || *count % 8 != 0) {
        return report("register count must be a multiple of 8 in [24, 256]");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateElectSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckElectSyncFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    mlir::Value membermask;
    if (!resolved_args.empty()) {
        membermask = castIntegerTo(builder, location, resolved_args[0],
                                   builder.getI32Type());
    }
    return mlir::NVVM::ElectSyncOp::create(builder, location,
                                           builder.getI1Type(), membermask)
        .getPred();
}

bool NVVMIntrinsic::CheckElectSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() > 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "elect_sync takes at most one membermask argument";
        return false;
    }
    if (!resolved_args.empty() &&
        (!resolved_args[0] || !resolved_args[0].getType().isIntOrIndex())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "elect_sync membermask must be an integer";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateNamedBarrierFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool arrive) const {
    llvm::StringRef name =
        arrive ? "named_barrier_arrive" : "named_barrier_sync";
    if (!CheckNamedBarrierFunction(call_expr, ctx, resolved_args, name)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto barrierId = castIntegerTo(builder, location, resolved_args[0],
                                   builder.getI32Type());
    auto threadCount = castIntegerTo(builder, location, resolved_args[1],
                                     builder.getI32Type());
    mlir::LLVM::InlineAsmOp::create(
        builder, location, mlir::TypeRange{},
        mlir::ValueRange{barrierId, threadCount},
        arrive ? "bar.arrive $0, $1;" : "bar.sync $0, $1;",
        "r,r,~{memory}", /*hasSideEffects=*/true,
        /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckNamedBarrierFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, llvm::StringRef name) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << name << " " << message;
        return false;
    };
    if (resolved_args.size() != 2 || !resolved_args[0] || !resolved_args[1] ||
        !resolved_args[0].getType().isIntOrIndex() ||
        !resolved_args[1].getType().isIntOrIndex()) {
        return report("requires integer barrier_id and thread_count operands");
    }
    if (auto id = getConstantIntValue(resolved_args[0]);
        id && (*id < 0 || *id > 15)) {
        return report("barrier_id must be in [0, 15]");
    }
    if (auto count = getConstantIntValue(resolved_args[1]);
        count && (*count <= 0 || *count % 32 != 0)) {
        return report("thread_count must be a positive multiple of 32");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateSyncWarpFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckSyncWarpFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    if (resolved_args.empty()) {
        mlir::LLVM::InlineAsmOp::create(
            builder, location, mlir::TypeRange{}, mlir::ValueRange{},
            "bar.warp.sync 0xffffffff;", "~{memory}",
            /*hasSideEffects=*/true, /*isAlignStack=*/false,
            mlir::LLVM::tailcallkind::TailCallKind::None,
            mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
        return ctx->GetCurrentFunctionGenerator()
            ->GetExprGenerator()
            ->CreateVoidValue();
    }
    mlir::Value mask;
    mask = castIntegerTo(builder, location, resolved_args[0],
                         builder.getI32Type());
    mlir::NVVM::SyncWarpOp::create(builder, location, mask);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckSyncWarpFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() > 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "syncwarp takes at most one mask argument";
        return false;
    }
    if (!resolved_args.empty() &&
        (!resolved_args[0] || !resolved_args[0].getType().isIntOrIndex())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "syncwarp mask must be an integer";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateClusterSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, llvm::StringRef name) const {
    if (!CheckNoArgumentFunction(call_expr, ctx, resolved_args, name)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    if (name == "fence_mbarrier_init_release_cluster") {
        mlir::NVVM::FenceMbarrierInitOp::create(builder, location);
    } else if (name == "cluster_arrive_relaxed") {
        mlir::NVVM::ClusterArriveRelaxedOp::create(
            builder, location, builder.getUnitAttr());
    } else {
        mlir::NVVM::ClusterWaitOp::create(builder, location,
                                          builder.getUnitAttr());
    }
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckNoArgumentFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, llvm::StringRef name) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << name << " requires no arguments";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateClusterBlockRankFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckNoArgumentFunction(call_expr, ctx, resolved_args,
                                 "cluster_block_rank")) {
        return nullptr;
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    return mlir::NVVM::ClusterId::create(builder, builder.getUnknownLoc(),
                                         builder.getI32Type())
        .getRes();
}

mlir::Value NVVMIntrinsic::CreateGriddepcontrolWaitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckNoArgumentFunction(call_expr, ctx, resolved_args,
                                 "griddepcontrol_wait")) {
        return nullptr;
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    mlir::NVVM::GriddepcontrolOp::create(
        builder, builder.getUnknownLoc(),
        mlir::NVVM::GridDepActionKind::wait);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value NVVMIntrinsic::CreateFenceProxyAsyncSharedCTAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    emitInlinePtxVoid(builder, location, "fence.proxy.async.shared::cta;");
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckFenceProxyAsyncSharedCTAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "fence_proxy_async_shared_cta requires no arguments";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateMakeWGMMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool returnRawBits) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMakeWGMMADescriptorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto substrateMemrefType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    auto workgroupMemorySpace = mlir::IntegerAttr::get(
        mlir::IntegerType::get(builder.getContext(), 64), 3);
    auto memrefType = mlir::MemRefType::getChecked(
        [&]() { return mlir::emitError(location); },
        substrateMemrefType.getShape(), substrateMemrefType.getElementType(),
        mlir::MemRefLayoutAttrInterface(), workgroupMemorySpace);
    if (!memrefType) {
        return nullptr;
    }

    mlir::Type resultType = builder.getI64Type();
    if (!returnRawBits) {
        resultType = mlir::nvgpu::WarpgroupMatrixDescriptorType::get(
            builder.getContext(), memrefType);
    }
    auto getI32Attr = [&](mlir::Value value) {
        return mlir::IntegerAttr::get(builder.getI32Type(),
                                      *ConstantFolder::FoldIntValue(value));
    };

    auto descriptor = cf::NVVMWGMMADescriptorOp::create(
        builder, location, resultType, resolved_args[0],
        getI32Attr(resolved_args[1]), getI32Attr(resolved_args[2]),
        getI32Attr(resolved_args[3]), getI32Attr(resolved_args[4]));
    mlir::Value result = descriptor.getResult();
    if (returnRawBits && resolved_args.size() == 7) {
        static constexpr int64_t kSwizzleBytes[] = {1, 32, 64, 128};
        int64_t swizzle = *getConstantIntValue(resolved_args[1]);
        int64_t layoutBytes = kSwizzleBytes[swizzle];
        int64_t defaultLeading =
            substrateMemrefType.getDimSize(0) * layoutBytes / 16;
        int64_t defaultStride = layoutBytes * 8 / 16;
        int64_t leading = *getConstantIntValue(resolved_args[5]) / 16;
        int64_t stride = *getConstantIntValue(resolved_args[6]) / 16;
        int64_t adjustment = (leading - defaultLeading) * (int64_t{1} << 16) +
                             (stride - defaultStride) * (int64_t{1} << 32);
        auto adjustmentValue = mlir::arith::ConstantIntOp::create(
            builder, location, adjustment, 64);
        result = mlir::arith::AddIOp::create(builder, location, result,
                                             adjustmentValue);
    }
    return result;
}

bool NVVMIntrinsic::CheckMakeWGMMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 5 && resolved_args.size() != 7) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor requires tensor, swizzle_kind, "
               "l2promo_kind, oob_kind, interleave_kind, and optional "
               "leading/stride byte offsets";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate tensor operand for make_wgmma_descriptor";
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor expects memref pointer operand for "
               "tensor";
        return false;
    }

    auto checkKind = [&](size_t index, llvm::StringRef name, int64_t maxValue) {
        if (!resolved_args[index]) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name
                << " operand for make_wgmma_descriptor";
            return false;
        }
        if (!resolved_args[index].getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_wgmma_descriptor " << name
                << " operand must be an integer type";
            return false;
        }
        auto value = ConstantFolder::FoldIntValue(resolved_args[index]);
        if (!value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_wgmma_descriptor requires a constant integer value "
                   "for "
                << name;
            return false;
        }
        if (*value < 0 || *value > maxValue) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_wgmma_descriptor " << name
                << " operand has invalid value " << *value;
            return false;
        }
        return true;
    };

    if (!checkKind(1, "swizzle_kind", 3) || !checkKind(2, "l2promo_kind", 3) ||
        !checkKind(3, "oob_kind", 1) || !checkKind(4, "interleave_kind", 2)) {
        return false;
    }
    if (resolved_args.size() == 7) {
        for (size_t index : {5u, 6u}) {
            auto value = getConstantIntValue(resolved_args[index]);
            if (!value || *value < 0 || *value > 0xffff0 ||
                (*value % 16) != 0) {
                ctx->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    call_expr->GetSourceRange().getBegin())
                    << "make_wgmma_descriptor leading/stride byte offsets "
                       "must be constant non-negative multiples of 16";
                return false;
            }
        }
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaInitResultFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckWgmmaInitResultFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto resultType = mlir::VectorType::get(
        {*getConstantIntValue(resolved_args[0])}, builder.getF32Type());
    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    return mlir::vector::BroadcastOp::create(builder, location, resultType,
                                             zero);
}

bool NVVMIntrinsic::CheckWgmmaInitResultFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1 || !resolved_args[0] ||
        !resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_init_result requires one constant integer size";
        return false;
    }
    auto size = getConstantIntValue(resolved_args[0]);
    if (!size || *size <= 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_init_result size must be a positive constant integer";
        return false;
    }
    return true;
}

bool NVVMIntrinsic::CheckRawWgmmaFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int64_t accumulatorSize,
    llvm::StringRef name) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << name << " " << message;
        return false;
    };
    if (resolved_args.size() != 4) {
        return report("requires desc_a, desc_b, accumulator, and scale_d");
    }
    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3]) {
        return report("received an invalid operand");
    }
    if (!resolved_args[0].getType().isInteger(64) ||
        !resolved_args[1].getType().isInteger(64)) {
        return report("descriptor operands must be i64 descriptor bits");
    }
    auto accumulatorType =
        mlir::dyn_cast<mlir::VectorType>(resolved_args[2].getType());
    if (!accumulatorType || accumulatorType.getRank() != 1 ||
        accumulatorType.getNumElements() != accumulatorSize ||
        !accumulatorType.getElementType().isF32()) {
        return report("received an incompatible f32 accumulator vector");
    }
    if (!resolved_args[3].getType().isInteger(32)) {
        return report("scale_d must be an i32 value");
    }
    auto scaleD = getConstantIntValue(resolved_args[3]);
    if (scaleD && *scaleD != 0 && *scaleD != 1) {
        return report("constant scale_d must be 0 or 1");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaM64NK16F32BF16BF16Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int64_t n) const {
    int64_t accumulatorSize = n / 2;
    std::string name = "wgmma_m64n" + std::to_string(n) + "k16_f32_bf16_bf16";
    if (!CheckRawWgmmaFunction(call_expr, ctx, resolved_args, accumulatorSize,
                               name)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Type> elementTypes(accumulatorSize,
                                               builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    llvm::SmallVector<mlir::Value> operands{resolved_args[0], resolved_args[1]};
    llvm::SmallVector<std::string> constraints(accumulatorSize, "=f");
    constraints.push_back("l");
    constraints.push_back("l");
    auto scaleD = getConstantIntValue(resolved_args[3]);
    bool dynamicScale = !scaleD;
    bool accumulate = dynamicScale || *scaleD == 1;
    if (accumulate) {
        for (int64_t i = 0; i < accumulatorSize; ++i) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, resolved_args[2], i));
            constraints.push_back(std::to_string(i));
        }
    }
    if (dynamicScale) {
        operands.push_back(resolved_args[3]);
        constraints.push_back("r");
    }

    std::string asmString;
    if (dynamicScale) {
        asmString = "{\n.reg .pred scale_d_pred;\nsetp.ne.b32 "
                    "scale_d_pred, $" +
                    std::to_string(accumulatorSize * 2 + 2) + ", 0;\n";
    }
    asmString += "wgmma.mma_async.sync.aligned.m64n" +
                            std::to_string(n) + "k16.f32.bf16.bf16 {";
    for (int64_t i = 0; i < accumulatorSize; ++i) {
        if (i != 0)
            asmString += ", ";
        asmString += "$" + std::to_string(i);
    }
    asmString += "}, $" + std::to_string(accumulatorSize) + ", $" +
                 std::to_string(accumulatorSize + 1) + ", ";
    if (dynamicScale) {
        asmString += "scale_d_pred, 1, 1, 0, 0;\n}";
    } else {
    asmString += accumulate ? "1, 1, 1, 0, 0;" : "0, 1, 1, 0, 0;";
    }

    std::string constraintString;
    for (size_t i = 0; i < constraints.size(); ++i) {
        if (i != 0)
            constraintString += ",";
        constraintString += constraints[i];
    }
    auto result = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, operands, asmString, constraintString,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});

    auto resultType = mlir::cast<mlir::VectorType>(resolved_args[2].getType());
    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value resultVector =
        mlir::vector::BroadcastOp::create(builder, location, resultType, zero);
    for (int64_t i = 0; i < accumulatorSize; ++i) {
        auto element = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), result.getRes(),
            llvm::ArrayRef<int64_t>{i});
        resultVector = mlir::vector::InsertOp::create(builder, location,
                                                      element, resultVector, i);
    }
    return resultVector;
}

bool NVVMIntrinsic::CheckWgmmaM64N128K128F32BF16BF16Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool registerShared, int64_t n,
    int64_t k) const {
    std::string name =
        registerShared
            ? "wgmma_m64n128k" + std::to_string(k) + "_f32_bf16_bf16_rs"
            : "wgmma_m64n" + std::to_string(n) + "k128_f32_bf16_bf16_ss";
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << name << " " << message;
        return false;
    };
    size_t registerTiles = k / 16;
    size_t expected = registerShared ? registerTiles + 2 : 2;
    if (resolved_args.size() != expected)
        return report(
            registerShared
                ? "requires one A fragment per K16 tile, desc_b, and acc"
                          : "requires desc_a and desc_b");
    if (registerShared) {
        for (size_t i = 0; i < registerTiles; ++i) {
            auto type =
                mlir::dyn_cast<mlir::VectorType>(resolved_args[i].getType());
            if (!type || type.getNumElements() != 4 ||
                !type.getElementType().isInteger(32))
                return report("A fragments must be vector<4xi32>");
        }
        if (!resolved_args[registerTiles].getType().isInteger(64))
            return report("descriptor must be i64");
        auto accType = mlir::dyn_cast<mlir::VectorType>(
            resolved_args[registerTiles + 1].getType());
        if (!accType || accType.getNumElements() != 64 ||
            !accType.getElementType().isF32())
            return report("accumulator must be vector<64xf32>");
    } else if (!resolved_args[0].getType().isInteger(64) ||
               !resolved_args[1].getType().isInteger(64)) {
        return report("descriptors must be i64");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaM64N128K128F32BF16BF16Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool registerShared, int64_t n,
    int64_t k) const {
    if (!CheckWgmmaM64N128K128F32BF16BF16Function(call_expr, ctx, resolved_args,
                                                  registerShared, n, k))
        return nullptr;

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    int64_t kOutputs = registerShared ? 64 : n / 2;
    llvm::SmallVector<mlir::Type> elementTypes(kOutputs, builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    llvm::SmallVector<mlir::Value> operands;
    llvm::SmallVector<std::string> constraints(kOutputs, "=f");
    std::string asmString;
    auto appendOutputs = [&]() {
        asmString += "{";
        for (int64_t i = 0; i < kOutputs; ++i) {
            if (i)
                asmString += ", ";
            asmString += "$" + std::to_string(i);
        }
        asmString += "}";
    };
    if (!registerShared) {
        mlir::Value descA = resolved_args[0];
        mlir::Value descB = resolved_args[1];
        for (int64_t tile = 0; tile < 8; ++tile) {
            operands.push_back(descA);
            operands.push_back(descB);
            constraints.push_back("l");
            constraints.push_back("l");
            if (tile < 7) {
                int64_t aStep = tile == 3 ? 506 : 2;
                int64_t bStep = tile == 3 ? n * 8 - 6 : 2;
                auto aOffset = mlir::arith::ConstantIntOp::create(
                    builder, location, aStep, 64);
                auto bOffset = mlir::arith::ConstantIntOp::create(
                    builder, location, bStep, 64);
                descA = mlir::arith::AddIOp::create(builder, location, descA,
                                                    aOffset);
                descB = mlir::arith::AddIOp::create(builder, location, descB,
                                                    bOffset);
            }
        }
        for (int64_t tile = 0; tile < 8; ++tile) {
            asmString += "wgmma.mma_async.sync.aligned.m64n" +
                         std::to_string(n) + "k16.f32.bf16.bf16 ";
            appendOutputs();
            int64_t operand = kOutputs + tile * 2;
            asmString += ", $" + std::to_string(operand) + ", $" +
                         std::to_string(operand + 1) +
                         (tile == 0 ? ", 0, 1, 1, 0, 0;\n"
                                    : ", 1, 1, 1, 0, 0;\n");
        }
    } else {
        int64_t registerTiles = k / 16;
        for (int64_t tile = 0; tile < registerTiles; ++tile) {
            for (int64_t i = 0; i < 4; ++i) {
                operands.push_back(mlir::vector::ExtractOp::create(
                    builder, location, resolved_args[tile], i));
                constraints.push_back("r");
            }
        }
        mlir::Value descB = resolved_args[registerTiles];
        for (int64_t tile = 0; tile < registerTiles; ++tile) {
            operands.push_back(descB);
            constraints.push_back("l");
            if (tile + 1 < registerTiles) {
                auto offset = mlir::arith::ConstantIntOp::create(
                    builder, location, 256, 64);
                descB = mlir::arith::AddIOp::create(builder, location, descB,
                                                    offset);
            }
        }
        for (int64_t i = 0; i < kOutputs; ++i) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, resolved_args[registerTiles + 1], i));
            constraints.push_back(std::to_string(i));
        }
        for (int64_t tile = 0; tile < registerTiles; ++tile) {
            asmString +=
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 ";
            appendOutputs();
            asmString += ", {";
            for (int64_t i = 0; i < 4; ++i) {
                if (i)
                    asmString += ", ";
                asmString += "$" + std::to_string(kOutputs + tile * 4 + i);
            }
            asmString += "}, $" +
                         std::to_string(kOutputs + registerTiles * 4 + tile) +
                         ", 1, 1, 1, 1;\n";
        }
    }

    std::string constraintString;
    for (size_t i = 0; i < constraints.size(); ++i) {
        if (i)
            constraintString += ",";
        constraintString += constraints[i];
    }
    auto result = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, operands, asmString, constraintString,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    auto resultType = mlir::VectorType::get({kOutputs}, builder.getF32Type());
    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value resultVector =
        mlir::vector::BroadcastOp::create(builder, location, resultType, zero);
    for (int64_t i = 0; i < kOutputs; ++i) {
        auto element = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), result.getRes(),
            llvm::ArrayRef<int64_t>{i});
        resultVector = mlir::vector::InsertOp::create(builder, location,
                                                      element, resultVector, i);
    }
    return resultVector;
}

bool NVVMIntrinsic::CheckWgmmaM64NK16F32BF16BF16RSFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int64_t n,
    llvm::StringRef name) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << name << " " << message;
        return false;
    };
    if (resolved_args.size() != 4) {
        return report("requires A registers, desc_b, accumulator, and scale_d");
    }
    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3]) {
        return report("received an invalid operand");
    }
    auto isVectorLike = [](mlir::Value value, int64_t elements,
                           mlir::Type elementType) {
        if (auto vectorType =
                mlir::dyn_cast<mlir::VectorType>(value.getType())) {
            return vectorType.getRank() == 1 &&
                   vectorType.getNumElements() == elements &&
                   vectorType.getElementType() == elementType;
        }
        if (auto memrefType =
                mlir::dyn_cast<cf::MemRefType>(value.getType())) {
            return memrefType.getRank() == 1 &&
                   memrefType.getShape()[0] == elements &&
                   memrefType.getElementType() == elementType;
        }
        return false;
    };
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    if (!isVectorLike(resolved_args[0], 4, builder.getI32Type())) {
        return report("A must be vector<4xi32> containing packed BF16 values");
    }
    if (!resolved_args[1].getType().isInteger(64)) {
        return report("desc_b must be i64 descriptor bits");
    }
    if (!isVectorLike(resolved_args[2], n / 2, builder.getF32Type())) {
        return report("accumulator must be a one-dimensional f32 vector");
    }
    auto scaleD = getConstantIntValue(resolved_args[3]);
    if (!scaleD || (*scaleD != 0 && *scaleD != 1)) {
        return report("scale_d must be the constant integer 0 or 1");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaM64NK16F32BF16BF16RSFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int64_t n) const {
    std::string name =
        "wgmma_m64n" + std::to_string(n) + "k16_f32_bf16_bf16_rs";
    if (!CheckWgmmaM64NK16F32BF16BF16RSFunction(
            call_expr, ctx, resolved_args, n, name)) {
        return nullptr;
    }

    const int64_t accumulatorSize = n / 2;
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Value> args(resolved_args.begin(),
                                        resolved_args.end());
    auto materializeVector = [&](mlir::Value value, int64_t elements,
                                 mlir::Type elementType) -> mlir::Value {
        if (mlir::isa<mlir::VectorType>(value.getType())) {
            return value;
        }
        auto memrefType = mlir::cast<cf::MemRefType>(value.getType());
        auto vectorType = mlir::VectorType::get({elements}, elementType);
        llvm::SmallVector<mlir::Value> indices;
        indices.reserve(memrefType.getRank());
        for (int64_t i = 0; i < memrefType.getRank(); ++i) {
            indices.push_back(mlir::arith::ConstantIndexOp::create(
                builder, location, 0));
        }
        return cf::AveLangMemRefLoadVecOp::create(
                   builder, location, vectorType, value, indices)
            .getResult();
    };
    args[0] = materializeVector(args[0], 4, builder.getI32Type());
    args[2] = materializeVector(args[2], accumulatorSize,
                                builder.getF32Type());
    llvm::SmallVector<mlir::Type> elementTypes(accumulatorSize,
                                               builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);

    llvm::SmallVector<mlir::Value> operands;
    for (int64_t i = 0; i < 4; ++i) {
        operands.push_back(mlir::vector::ExtractOp::create(
            builder, location, args[0], i));
    }
    operands.push_back(args[1]);
    llvm::SmallVector<std::string> constraints(accumulatorSize, "=f");
    constraints.insert(constraints.end(), 4, "r");
    constraints.push_back("l");
    bool accumulate = *getConstantIntValue(args[3]) == 1;
    if (accumulate) {
        for (int64_t i = 0; i < accumulatorSize; ++i) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, args[2], i));
            constraints.push_back(std::to_string(i));
        }
    }

    std::string asmString =
        "wgmma.mma_async.sync.aligned.m64n" + std::to_string(n) +
        "k16.f32.bf16.bf16 {";
    for (int64_t i = 0; i < accumulatorSize; ++i) {
        if (i != 0)
            asmString += ", ";
        asmString += "$" + std::to_string(i);
    }
    asmString += "}, {$" + std::to_string(accumulatorSize) + ", $" +
                 std::to_string(accumulatorSize + 1) + ", $" +
                 std::to_string(accumulatorSize + 2) + ", $" +
                 std::to_string(accumulatorSize + 3) + "}, $" +
                 std::to_string(accumulatorSize + 4) + ", ";
    asmString += accumulate ? "1, 1, 1, 1;" : "0, 1, 1, 1;";

    std::string constraintString;
    for (size_t i = 0; i < constraints.size(); ++i) {
        if (i != 0)
            constraintString += ",";
        constraintString += constraints[i];
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, operands, asmString, constraintString,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});

    auto resultType = mlir::cast<mlir::VectorType>(args[2].getType());
    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value resultVector = mlir::vector::BroadcastOp::create(
        builder, location, resultType, zero);
    for (int64_t i = 0; i < accumulatorSize; ++i) {
        auto element = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{i});
        resultVector = mlir::vector::InsertOp::create(
            builder, location, element, resultVector, i);
    }
    return resultVector;
}

bool NVVMIntrinsic::CheckWgmmaM64N16K16F32BF16BF16RSX8Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    constexpr llvm::StringLiteral kName =
        "wgmma_m64n16k16_f32_bf16_bf16_rs_x8";
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << kName << " " << message;
        return false;
    };
    if (resolved_args.size() != 5 ||
        llvm::any_of(resolved_args, [](mlir::Value value) { return !value; })) {
        return report(
            "requires A[32], desc_b, accumulator, scale_d, and descriptor step");
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto aType = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!aType || aType.getRank() != 1 || aType.getShape()[0] != 32 ||
        aType.getElementType() != builder.getI32Type()) {
        return report("A must be a 32-element i32 tensor of packed BF16 fragments");
    }
    if (!resolved_args[1].getType().isInteger(64)) {
        return report("desc_b must be i64 descriptor bits");
    }
    auto isAccumulator = [&](mlir::Value value) {
        if (auto vectorType =
                mlir::dyn_cast<mlir::VectorType>(value.getType())) {
            return vectorType.getRank() == 1 &&
                   vectorType.getNumElements() == 8 &&
                   vectorType.getElementType().isF32();
        }
        if (auto memrefType =
                mlir::dyn_cast<cf::MemRefType>(value.getType())) {
            return memrefType.getRank() == 1 &&
                   memrefType.getShape()[0] == 8 &&
                   memrefType.getElementType().isF32();
        }
        return false;
    };
    if (!isAccumulator(resolved_args[2])) {
        return report("accumulator must be a one-dimensional 8xf32 value");
    }
    auto scaleD = getConstantIntValue(resolved_args[3]);
    if (!scaleD || (*scaleD != 0 && *scaleD != 1)) {
        return report("scale_d must be the constant integer 0 or 1");
    }
    auto descriptorStep = getConstantIntValue(resolved_args[4]);
    if (!descriptorStep || *descriptorStep < 0) {
        return report("descriptor step must be a non-negative constant integer");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaM64N16K16F32BF16BF16RSX8Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckWgmmaM64N16K16F32BF16BF16RSX8Function(
            call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    constexpr int64_t kTiles = 8;
    constexpr int64_t kARegisters = 4;
    constexpr int64_t kOutputs = 8;
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto vectorI32x4 = mlir::VectorType::get({kARegisters},
                                             builder.getI32Type());
    auto resultType = mlir::VectorType::get({kOutputs}, builder.getF32Type());

    mlir::Value accumulator = resolved_args[2];
    if (mlir::isa<cf::MemRefType>(accumulator.getType())) {
        auto zero = mlir::arith::ConstantIndexOp::create(builder, location, 0);
        accumulator = cf::AveLangMemRefLoadVecOp::create(
                          builder, location, resultType, accumulator,
                          mlir::ValueRange{zero})
                          .getResult();
    }

    llvm::SmallVector<mlir::Value> aFragments;
    aFragments.reserve(kTiles);
    for (int64_t tile = 0; tile < kTiles; ++tile) {
        auto offset = mlir::arith::ConstantIndexOp::create(
            builder, location, tile * kARegisters);
        aFragments.push_back(cf::AveLangMemRefLoadVecOp::create(
                                 builder, location, vectorI32x4,
                                 resolved_args[0], mlir::ValueRange{offset})
                                 .getResult());
    }

    const int64_t descriptorStep =
        *getConstantIntValue(resolved_args[4]);

    llvm::SmallVector<mlir::Type> elementTypes(kOutputs,
                                               builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    llvm::SmallVector<mlir::Value> operands;
    for (auto fragment : aFragments) {
        for (int64_t element = 0; element < kARegisters; ++element) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, fragment, element));
        }
    }
    for (int64_t tile = 0; tile < kTiles; ++tile) {
        if (tile == 0 || descriptorStep == 0) {
            operands.push_back(resolved_args[1]);
        } else {
            auto offset = mlir::arith::ConstantIntOp::create(
                builder, location, tile * descriptorStep, 64);
            operands.push_back(mlir::arith::AddIOp::create(
                builder, location, resolved_args[1], offset));
        }
    }

    llvm::SmallVector<std::string> constraints(kOutputs, "=f");
    constraints.insert(constraints.end(), kTiles * kARegisters, "r");
    constraints.insert(constraints.end(), kTiles, "l");
    const bool accumulate = *getConstantIntValue(resolved_args[3]) == 1;
    if (accumulate) {
        for (int64_t output = 0; output < kOutputs; ++output) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, accumulator, output));
            constraints.push_back(std::to_string(output));
        }
    }

    const int64_t descriptorBase = kOutputs + kTiles * kARegisters;
    std::string asmString = "";
    for (int64_t tile = 0; tile < kTiles; ++tile) {
        asmString +=
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {";
        for (int64_t output = 0; output < kOutputs; ++output) {
            if (output != 0)
                asmString += ", ";
            asmString += "$" + std::to_string(output);
        }
        const int64_t aBase = kOutputs + tile * kARegisters;
        asmString += "}, {" + std::string("$") + std::to_string(aBase) +
                     ", $" + std::to_string(aBase + 1) + ", $" +
                     std::to_string(aBase + 2) + ", $" +
                     std::to_string(aBase + 3) + "}, $" +
                     std::to_string(descriptorBase + tile) + ", " +
                     std::to_string(tile == 0 && !accumulate ? 0 : 1) +
                     ", 1, 1, 0;\n";
    }

    std::string constraintString;
    for (size_t i = 0; i < constraints.size(); ++i) {
        if (i != 0)
            constraintString += ",";
        constraintString += constraints[i];
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, operands, asmString, constraintString,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});

    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value result =
        mlir::vector::BroadcastOp::create(builder, location, resultType, zero);
    for (int64_t output = 0; output < kOutputs; ++output) {
        auto element = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{output});
        result = mlir::vector::InsertOp::create(builder, location, element,
                                                result, output);
    }
    return result;
}

bool NVVMIntrinsic::CheckWgmmaM64N16K16F32BF16BF16RSSmemX8Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    constexpr llvm::StringLiteral kName =
        "wgmma_m64n16k16_f32_bf16_bf16_rs_smem_x8";
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << kName << " " << message;
        return false;
    };
    if (resolved_args.size() != 9 ||
        llvm::any_of(resolved_args, [](mlir::Value value) { return !value; })) {
        return report("requires A[32], shared-memory base/stage, accumulator, "
                      "scale_d, byte layout, and descriptor step");
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto aType = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!aType || aType.getRank() != 1 || aType.getShape()[0] != 32 ||
        aType.getElementType() != builder.getI32Type()) {
        return report("A must be a 32-element i32 tensor of packed BF16 fragments");
    }
    if (!resolved_args[1].getType().isInteger(32) ||
        !resolved_args[2].getType().isInteger(32)) {
        return report("shared-memory base and stage must be i32 values");
    }
    auto isAccumulator = [&](mlir::Value value) {
        if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(value.getType())) {
            return vectorType.getRank() == 1 &&
                   vectorType.getNumElements() == 8 &&
                   vectorType.getElementType().isF32();
        }
        if (auto memrefType = mlir::dyn_cast<cf::MemRefType>(value.getType())) {
            return memrefType.getRank() == 1 &&
                   memrefType.getShape()[0] == 8 &&
                   memrefType.getElementType().isF32();
        }
        return false;
    };
    if (!isAccumulator(resolved_args[3])) {
        return report("accumulator must be a one-dimensional 8xf32 value");
    }
    auto scaleD = getConstantIntValue(resolved_args[4]);
    if (!scaleD || (*scaleD != 0 && *scaleD != 1)) {
        return report("scale_d must be the constant integer 0 or 1");
    }
    for (size_t index = 5; index < 9; ++index) {
        auto value = getConstantIntValue(resolved_args[index]);
        if (!value || *value < 0) {
            return report("byte layout and descriptor step must be "
                          "non-negative constant integers");
        }
    }
    auto leading = *getConstantIntValue(resolved_args[7]);
    if ((leading % 16) != 0 || leading > 0xffff0) {
        return report("leading byte offset must be a multiple of 16");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaM64N16K16F32BF16BF16RSSmemX8Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckWgmmaM64N16K16F32BF16BF16RSSmemX8Function(
            call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    constexpr int64_t kTiles = 8;
    constexpr int64_t kARegisters = 4;
    constexpr int64_t kOutputs = 8;
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto vectorI32x4 = mlir::VectorType::get({kARegisters},
                                             builder.getI32Type());
    auto resultType = mlir::VectorType::get({kOutputs}, builder.getF32Type());

    mlir::Value accumulator = resolved_args[3];
    if (mlir::isa<cf::MemRefType>(accumulator.getType())) {
        auto zero = mlir::arith::ConstantIndexOp::create(builder, location, 0);
        accumulator = cf::AveLangMemRefLoadVecOp::create(
                          builder, location, resultType, accumulator,
                          mlir::ValueRange{zero})
                          .getResult();
    }

    llvm::SmallVector<mlir::Value> aFragments;
    aFragments.reserve(kTiles);
    for (int64_t tile = 0; tile < kTiles; ++tile) {
        auto offset = mlir::arith::ConstantIndexOp::create(
            builder, location, tile * kARegisters);
        aFragments.push_back(cf::AveLangMemRefLoadVecOp::create(
                                 builder, location, vectorI32x4,
                                 resolved_args[0], mlir::ValueRange{offset})
                                 .getResult());
    }

    llvm::SmallVector<mlir::Type> elementTypes(kOutputs,
                                               builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    llvm::SmallVector<mlir::Value> operands;
    for (auto fragment : aFragments) {
        for (int64_t element = 0; element < kARegisters; ++element) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, fragment, element));
        }
    }
    operands.push_back(resolved_args[1]);
    operands.push_back(resolved_args[2]);

    llvm::SmallVector<std::string> constraints(kOutputs, "=f");
    constraints.insert(constraints.end(), kTiles * kARegisters + 2, "r");
    const bool accumulate = *getConstantIntValue(resolved_args[4]) == 1;
    if (accumulate) {
        for (int64_t output = 0; output < kOutputs; ++output) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, accumulator, output));
            constraints.push_back(std::to_string(output));
        }
    }

    const int64_t baseByteOffset =
        *getConstantIntValue(resolved_args[5]);
    const int64_t stageByteStride =
        *getConstantIntValue(resolved_args[6]);
    const int64_t leadingByteOffset =
        *getConstantIntValue(resolved_args[7]);
    const int64_t descriptorStep =
        *getConstantIntValue(resolved_args[8]);
    constexpr int64_t kSmemBase = kOutputs + kTiles * kARegisters;
    constexpr int64_t kStage = kSmemBase + 1;
    std::string asmString =
        "{ .reg .b32 addr, bits; .reg .b64 desc;\n"
        "mad.lo.u32 addr, $" +
        std::to_string(kStage) + ", " + std::to_string(stageByteStride) +
        ", $" + std::to_string(kSmemBase) + ";\n";
    if (baseByteOffset != 0) {
        asmString += "add.u32 addr, addr, " +
                     std::to_string(baseByteOffset) + ";\n";
    }
    asmString +=
        "and.b32 bits, addr, 0x3ffff;\n"
        "shr.u32 bits, bits, 4;\n"
        "or.b32 bits, bits, " +
        std::to_string((leadingByteOffset / 16) << 16) +
        ";\ncvt.u64.u32 desc, bits;\n"
        "or.b64 desc, desc, 34359738368;\n";
    for (int64_t tile = 0; tile < kTiles; ++tile) {
        if (tile != 0 && descriptorStep != 0) {
            asmString += "add.u64 desc, desc, " +
                         std::to_string(descriptorStep) + ";\n";
        }
        asmString +=
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {";
        for (int64_t output = 0; output < kOutputs; ++output) {
            if (output != 0)
                asmString += ", ";
            asmString += "$" + std::to_string(output);
        }
        const int64_t aBase = kOutputs + tile * kARegisters;
        asmString += "}, {$" + std::to_string(aBase) + ", $" +
                     std::to_string(aBase + 1) + ", $" +
                     std::to_string(aBase + 2) + ", $" +
                     std::to_string(aBase + 3) + "}, desc, " +
                     std::to_string(tile == 0 && !accumulate ? 0 : 1) +
                     ", 1, 1, 0;\n";
    }
    asmString += "}";

    std::string constraintString;
    for (size_t index = 0; index < constraints.size(); ++index) {
        if (index != 0)
            constraintString += ",";
        constraintString += constraints[index];
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, operands, asmString, constraintString,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});

    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value result =
        mlir::vector::BroadcastOp::create(builder, location, resultType, zero);
    for (int64_t output = 0; output < kOutputs; ++output) {
        auto element = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{output});
        result = mlir::vector::InsertOp::create(builder, location, element,
                                                result, output);
    }
    return result;
}

bool NVVMIntrinsic::CheckWgmmaM64N16K16F32BF16BF16RSF32X8Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    constexpr llvm::StringLiteral kName =
        "wgmma_m64n16k16_f32_bf16_bf16_rs_f32_x8";
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << kName << " " << message;
        return false;
    };
    if (resolved_args.size() != 6 ||
        llvm::any_of(resolved_args, [](mlir::Value value) { return !value; })) {
        return report("requires A0[32], A1[32], desc_b, accumulator, "
                      "scale_d, and descriptor step");
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto isF32Vector = [&](mlir::Value value, int64_t elements) {
        if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(value.getType())) {
            return vectorType.getRank() == 1 &&
                   vectorType.getNumElements() == elements &&
                   vectorType.getElementType().isF32();
        }
        if (auto memrefType = mlir::dyn_cast<cf::MemRefType>(value.getType())) {
            return memrefType.getRank() == 1 &&
                   memrefType.getShape()[0] == elements &&
                   memrefType.getElementType() == builder.getF32Type();
        }
        return false;
    };
    if (!isF32Vector(resolved_args[0], 32) ||
        !isF32Vector(resolved_args[1], 32)) {
        return report("A0 and A1 must each be one-dimensional 32xf32 values");
    }
    if (!resolved_args[2].getType().isInteger(64)) {
        return report("desc_b must be i64 descriptor bits");
    }
    if (!isF32Vector(resolved_args[3], 8)) {
        return report("accumulator must be a one-dimensional 8xf32 value");
    }
    auto scaleD = getConstantIntValue(resolved_args[4]);
    if (!scaleD || (*scaleD != 0 && *scaleD != 1)) {
        return report("scale_d must be the constant integer 0 or 1");
    }
    auto descriptorStep = getConstantIntValue(resolved_args[5]);
    if (!descriptorStep || *descriptorStep < 0) {
        return report("descriptor step must be a non-negative constant integer");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaM64N16K16F32BF16BF16RSF32X8Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckWgmmaM64N16K16F32BF16BF16RSF32X8Function(
            call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    constexpr int64_t kTiles = 8;
    constexpr int64_t kValuesPerTile = 8;
    constexpr int64_t kOutputs = 8;
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto vectorF32x32 =
        mlir::VectorType::get({32}, builder.getF32Type());
    auto resultType =
        mlir::VectorType::get({kOutputs}, builder.getF32Type());

    auto materializeVector = [&](mlir::Value value,
                                 mlir::VectorType type) -> mlir::Value {
        if (!mlir::isa<cf::MemRefType>(value.getType())) {
            return value;
        }
        auto zero = mlir::arith::ConstantIndexOp::create(builder, location, 0);
        return cf::AveLangMemRefLoadVecOp::create(
                   builder, location, type, value, mlir::ValueRange{zero})
            .getResult();
    };
    auto a0 = materializeVector(resolved_args[0], vectorF32x32);
    auto a1 = materializeVector(resolved_args[1], vectorF32x32);
    auto accumulator = materializeVector(resolved_args[3], resultType);

    llvm::SmallVector<mlir::Type> elementTypes(kOutputs,
                                               builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    llvm::SmallVector<mlir::Value> operands;
    operands.reserve(2 * 32 + 1 + kOutputs);
    for (auto source : {a0, a1}) {
        for (int64_t element = 0; element < 32; ++element) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, source, element));
        }
    }
    operands.push_back(resolved_args[2]);

    llvm::SmallVector<std::string> constraints(kOutputs, "=f");
    constraints.insert(constraints.end(), 2 * 32, "f");
    constraints.push_back("l");
    const bool accumulate = *getConstantIntValue(resolved_args[4]) == 1;
    if (accumulate) {
        for (int64_t output = 0; output < kOutputs; ++output) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, accumulator, output));
            constraints.push_back(std::to_string(output));
        }
    }

    const int64_t descriptorStep =
        *getConstantIntValue(resolved_args[5]);
    const int64_t inputBase = kOutputs;
    const int64_t descriptorOperand = inputBase + 2 * 32;
    std::string asmString =
        "{ .reg .b32 a<4>; .reg .b64 desc;\nmov.b64 desc, $" +
        std::to_string(descriptorOperand) + ";\n";
    for (int64_t tile = 0; tile < kTiles; ++tile) {
        if (tile != 0 && descriptorStep != 0) {
            asmString += "add.u64 desc, desc, " +
                         std::to_string(descriptorStep) + ";\n";
        }
        const int64_t sourceBase = inputBase + tile * kValuesPerTile;
        for (int64_t pair = 0; pair < 4; ++pair) {
            const int64_t lo = sourceBase + 2 * pair;
            const int64_t hi = lo + 1;
            asmString += "cvt.rn.bf16x2.f32 a" + std::to_string(pair) +
                         ", $" + std::to_string(hi) + ", $" +
                         std::to_string(lo) + ";\n";
        }
        asmString +=
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {";
        for (int64_t output = 0; output < kOutputs; ++output) {
            if (output != 0)
                asmString += ", ";
            asmString += "$" + std::to_string(output);
        }
        asmString += "}, {a0, a1, a2, a3}, desc, " +
                     std::to_string(tile == 0 && !accumulate ? 0 : 1) +
                     ", 1, 1, 0;\n";
    }
    asmString += "}";

    std::string constraintString;
    for (size_t index = 0; index < constraints.size(); ++index) {
        if (index != 0)
            constraintString += ",";
        constraintString += constraints[index];
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, operands, asmString, constraintString,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});

    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value result =
        mlir::vector::BroadcastOp::create(builder, location, resultType, zero);
    for (int64_t output = 0; output < kOutputs; ++output) {
        auto element = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{output});
        result = mlir::vector::InsertOp::create(builder, location, element,
                                                result, output);
    }
    return result;
}

bool NVVMIntrinsic::
    CheckWgmmaM64N16M64N64X2K16F32BF16BF16RSBranchFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const {
    constexpr llvm::StringLiteral kName =
        "wgmma_m64n16x2_m64n64x4_k16_f32_bf16_bf16_rs_branch";
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << kName << " " << message;
        return false;
    };
    if (resolved_args.size() != 11 ||
        llvm::any_of(resolved_args, [](mlir::Value value) { return !value; })) {
        return report("requires two A fragments, shared base/stage, "
                      "output/state accumulators, and zero_start");
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto isVectorLike = [](mlir::Value value, int64_t elements,
                           mlir::Type elementType) {
        if (auto vectorType =
                mlir::dyn_cast<mlir::VectorType>(value.getType())) {
            return vectorType.getRank() == 1 &&
                   vectorType.getNumElements() == elements &&
                   vectorType.getElementType() == elementType;
        }
        if (auto memrefType = mlir::dyn_cast<cf::MemRefType>(value.getType())) {
            return memrefType.getRank() == 1 &&
                   memrefType.getShape()[0] == elements &&
                   memrefType.getElementType() == elementType;
        }
        return false;
    };
    if (!isVectorLike(resolved_args[0], 4, builder.getI32Type()) ||
        !isVectorLike(resolved_args[1], 4, builder.getI32Type())) {
        return report("A fragments must be one-dimensional 4xi32 values");
    }
    if (!resolved_args[2].getType().isInteger(32) ||
        !resolved_args[3].getType().isInteger(32)) {
        return report("shared base and stage must be i32 values");
    }
    if (!isVectorLike(resolved_args[4], 8, builder.getF32Type()) ||
        !isVectorLike(resolved_args[5], 8, builder.getF32Type())) {
        return report("output accumulators must be one-dimensional 8xf32 values");
    }
    if (!isVectorLike(resolved_args[6], 32, builder.getF32Type()) ||
        !isVectorLike(resolved_args[7], 32, builder.getF32Type()) ||
        !isVectorLike(resolved_args[8], 32, builder.getF32Type()) ||
        !isVectorLike(resolved_args[9], 32, builder.getF32Type())) {
        return report("state accumulators must be one-dimensional 32xf32 values");
    }
    if (!resolved_args[10].getType().isInteger(32)) {
        return report("zero_start must be i32");
    }
    return true;
}

mlir::Value NVVMIntrinsic::
    CreateWgmmaM64N16M64N64X2K16F32BF16BF16RSBranchFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckWgmmaM64N16M64N64X2K16F32BF16BF16RSBranchFunction(
            call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    constexpr int64_t kOutputElements = 8;
    constexpr int64_t kStateElements = 32;
    constexpr int64_t kResultElements =
        2 * kOutputElements + 4 * kStateElements;
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto materializeVector = [&](mlir::Value value, int64_t elements,
                                 mlir::Type elementType) -> mlir::Value {
        if (mlir::isa<mlir::VectorType>(value.getType())) {
            return value;
        }
        auto vectorType = mlir::VectorType::get({elements}, elementType);
        auto zero = mlir::arith::ConstantIndexOp::create(builder, location, 0);
        return cf::AveLangMemRefLoadVecOp::create(
                   builder, location, vectorType, value,
                   mlir::ValueRange{zero})
            .getResult();
    };

    auto a0 = materializeVector(resolved_args[0], 4, builder.getI32Type());
    auto a1 = materializeVector(resolved_args[1], 4, builder.getI32Type());
    auto out0 = materializeVector(resolved_args[4], kOutputElements,
                                  builder.getF32Type());
    auto out1 = materializeVector(resolved_args[5], kOutputElements,
                                  builder.getF32Type());
    auto state0 = materializeVector(resolved_args[6], kStateElements,
                                    builder.getF32Type());
    auto state1 = materializeVector(resolved_args[7], kStateElements,
                                    builder.getF32Type());
    auto state2 = materializeVector(resolved_args[8], kStateElements,
                                    builder.getF32Type());
    auto state3 = materializeVector(resolved_args[9], kStateElements,
                                    builder.getF32Type());
    llvm::SmallVector<mlir::Type> elementTypes(kResultElements,
                                               builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    llvm::SmallVector<mlir::Value> operands;
    for (int64_t element = 0; element < 4; ++element) {
        operands.push_back(mlir::vector::ExtractOp::create(
            builder, location, a0, element));
    }
    for (int64_t element = 0; element < 4; ++element) {
        operands.push_back(mlir::vector::ExtractOp::create(
            builder, location, a1, element));
    }
    operands.push_back(resolved_args[2]);
    operands.push_back(resolved_args[3]);
    operands.push_back(resolved_args[10]);

    llvm::SmallVector<std::string> constraints(kResultElements, "=f");
    constraints.insert(constraints.end(), 8, "r");
    constraints.insert(constraints.end(), 3, "r");
    auto appendTiedInputs = [&](mlir::Value value, int64_t elements,
                                int64_t resultOffset) {
        for (int64_t element = 0; element < elements; ++element) {
            operands.push_back(mlir::vector::ExtractOp::create(
                builder, location, value, element));
            constraints.push_back(std::to_string(resultOffset + element));
        }
    };
    appendTiedInputs(out0, kOutputElements, 0);
    appendTiedInputs(out1, kOutputElements, kOutputElements);
    appendTiedInputs(state0, kStateElements, 2 * kOutputElements);
    appendTiedInputs(state1, kStateElements,
                     2 * kOutputElements + kStateElements);
    appendTiedInputs(state2, kStateElements,
                     2 * kOutputElements + 2 * kStateElements);
    appendTiedInputs(state3, kStateElements,
                     2 * kOutputElements + 3 * kStateElements);

    auto registerList = [](int64_t begin, int64_t count) {
        std::string result = "{";
        for (int64_t element = 0; element < count; ++element) {
            if (element != 0)
                result += ", ";
            result += "$" + std::to_string(begin + element);
        }
        return result + "}";
    };
    constexpr int64_t kA0Base = kResultElements;
    constexpr int64_t kA1Base = kA0Base + 4;
    constexpr int64_t kSmemBase = kA1Base + 4;
    constexpr int64_t kStage = kSmemBase + 1;
    constexpr int64_t kZeroStart = kStage + 1;
    constexpr int64_t kOut0 = 0;
    constexpr int64_t kOut1 = kOut0 + kOutputElements;
    constexpr int64_t kState0 = kOut1 + kOutputElements;
    constexpr int64_t kState1 = kState0 + kStateElements;
    constexpr int64_t kState2 = kState1 + kStateElements;
    constexpr int64_t kState3 = kState2 + kStateElements;
    const std::string a0List = registerList(kA0Base, 4);
    const std::string a1List = registerList(kA1Base, 4);
    std::string asmString =
        "{ .reg .pred p; .reg .b32 addr, bits; "
        ".reg .b64 d16, d64, d64hi;\n"
        "mad.lo.u32 addr, $" +
        std::to_string(kStage) + ", 512, $" +
        std::to_string(kSmemBase) + ";\n"
        "add.u32 addr, addr, 25088;\n"
        "and.b32 bits, addr, 0x3ffff;\n"
        "shr.u32 bits, bits, 4;\n"
        "or.b32 bits, bits, 1048576;\n"
        "cvt.u64.u32 d16, bits;\n"
        "or.b64 d16, d16, 34359738368;\n"
        "mad.lo.u32 addr, $" +
        std::to_string(kStage) + ", 4096, $" +
        std::to_string(kSmemBase) + ";\n"
        "add.u32 addr, addr, 16896;\n"
        "and.b32 bits, addr, 0x3ffff;\n"
        "shr.u32 bits, bits, 4;\n"
        "or.b32 bits, bits, 4194304;\n"
        "cvt.u64.u32 d64, bits;\n"
        "or.b64 d64, d64, 34359738368;\n"
        "add.u64 d64hi, d64, 128;\n"
        "setp.ne.b32 p, $" +
        std::to_string(kZeroStart) +
        ", 0;\n@p bra AVELANG_WGMMA_ZERO_${:uid};\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 " +
        registerList(kOut0, kOutputElements) + ", " + a0List +
        ", d16, 1, 1, 1, 0;\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " +
        registerList(kState0, kStateElements) + ", " + a0List +
        ", d64, 1, 1, 1, 0;\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " +
        registerList(kState1, kStateElements) + ", " + a0List +
        ", d64hi, 1, 1, 1, 0;\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 " +
        registerList(kOut1, kOutputElements) + ", " + a1List +
        ", d16, 1, 1, 1, 0;\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " +
        registerList(kState2, kStateElements) + ", " + a1List +
        ", d64, 1, 1, 1, 0;\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " +
        registerList(kState3, kStateElements) + ", " + a1List +
        ", d64hi, 1, 1, 1, 0;\n"
        "bra AVELANG_WGMMA_DONE_${:uid};\n"
        "AVELANG_WGMMA_ZERO_${:uid}:\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " +
        registerList(kState0, kStateElements) + ", " + a0List +
        ", d64, 0, 1, 1, 0;\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " +
        registerList(kState1, kStateElements) + ", " + a0List +
        ", d64hi, 0, 1, 1, 0;\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " +
        registerList(kState2, kStateElements) + ", " + a1List +
        ", d64, 0, 1, 1, 0;\n";
    asmString +=
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " +
        registerList(kState3, kStateElements) + ", " + a1List +
        ", d64hi, 0, 1, 1, 0;\n"
        "AVELANG_WGMMA_DONE_${:uid}:\n}";

    std::string constraintString;
    for (size_t index = 0; index < constraints.size(); ++index) {
        if (index != 0)
            constraintString += ",";
        constraintString += constraints[index];
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, operands, asmString, constraintString,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});

    auto resultType = mlir::VectorType::get({kResultElements},
                                            builder.getF32Type());
    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value result = mlir::vector::BroadcastOp::create(
        builder, location, resultType, zero);
    for (int64_t element = 0; element < kResultElements; ++element) {
        auto value = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{element});
        result = mlir::vector::InsertOp::create(
            builder, location, value, result, element);
    }
    return result;
}

bool NVVMIntrinsic::CheckWgmmaM64NK16F32BF16BF16RSPredicatedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int64_t n,
    llvm::StringRef name) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << name << " " << message;
        return false;
    };
    if (resolved_args.size() != 5 ||
        llvm::any_of(resolved_args, [](mlir::Value value) { return !value; })) {
        return report(
            "requires A registers, desc_b, accumulator, scale_d, and execute");
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto isVectorLike = [](mlir::Value value, int64_t elements,
                           mlir::Type elementType) {
        if (auto vectorType =
                mlir::dyn_cast<mlir::VectorType>(value.getType())) {
            return vectorType.getRank() == 1 &&
                   vectorType.getNumElements() == elements &&
                   vectorType.getElementType() == elementType;
        }
        if (auto memrefType =
                mlir::dyn_cast<cf::MemRefType>(value.getType())) {
            return memrefType.getRank() == 1 &&
                   memrefType.getShape()[0] == elements &&
                   memrefType.getElementType() == elementType;
        }
        return false;
    };
    if (!isVectorLike(resolved_args[0], 4, builder.getI32Type())) {
        return report("A must be vector<4xi32> containing packed BF16 values");
    }
    if (!resolved_args[1].getType().isInteger(64)) {
        return report("desc_b must be i64 descriptor bits");
    }
    if (!isVectorLike(resolved_args[2], n / 2, builder.getF32Type())) {
        return report("accumulator has the wrong f32 vector width");
    }
    auto scaleD = getConstantIntValue(resolved_args[3]);
    if (!scaleD || (*scaleD != 0 && *scaleD != 1)) {
        return report("scale_d must be the constant integer 0 or 1");
    }
    if (!resolved_args[4].getType().isInteger(32)) {
        return report("execute must be i32");
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaM64NK16F32BF16BF16RSPredicatedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int64_t n) const {
    std::string name = "wgmma_m64n" + std::to_string(n) +
                       "k16_f32_bf16_bf16_rs_predicated";
    if (!CheckWgmmaM64NK16F32BF16BF16RSPredicatedFunction(
            call_expr, ctx, resolved_args, n, name)) {
        return nullptr;
    }

    const int64_t accumulatorSize = n / 2;
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Value> args(resolved_args.begin(),
                                        resolved_args.end());
    auto materializeVector = [&](mlir::Value value, int64_t elements,
                                 mlir::Type elementType) -> mlir::Value {
        if (mlir::isa<mlir::VectorType>(value.getType())) {
            return value;
        }
        auto memrefType = mlir::cast<cf::MemRefType>(value.getType());
        auto vectorType = mlir::VectorType::get({elements}, elementType);
        llvm::SmallVector<mlir::Value> indices;
        for (int64_t i = 0; i < memrefType.getRank(); ++i) {
            indices.push_back(mlir::arith::ConstantIndexOp::create(
                builder, location, 0));
        }
        return cf::AveLangMemRefLoadVecOp::create(
                   builder, location, vectorType, value, indices)
            .getResult();
    };
    args[0] = materializeVector(args[0], 4, builder.getI32Type());
    args[2] = materializeVector(args[2], accumulatorSize,
                                builder.getF32Type());

    llvm::SmallVector<mlir::Type> elementTypes(accumulatorSize,
                                               builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    llvm::SmallVector<mlir::Value> operands;
    for (int64_t i = 0; i < 4; ++i) {
        operands.push_back(mlir::vector::ExtractOp::create(
            builder, location, args[0], i));
    }
    operands.push_back(args[1]);
    operands.push_back(args[4]);
    llvm::SmallVector<std::string> constraints(accumulatorSize, "=f");
    constraints.insert(constraints.end(), 4, "r");
    constraints.push_back("l");
    constraints.push_back("r");
    for (int64_t i = 0; i < accumulatorSize; ++i) {
        operands.push_back(mlir::vector::ExtractOp::create(
            builder, location, args[2], i));
        constraints.push_back(std::to_string(i));
    }

    std::string asmString = "{ .reg .pred p; setp.ne.b32 p, $" +
                            std::to_string(accumulatorSize + 5) + ", 0;\n@p " +
                            "wgmma.mma_async.sync.aligned.m64n" +
                            std::to_string(n) +
                            "k16.f32.bf16.bf16 {";
    for (int64_t i = 0; i < accumulatorSize; ++i) {
        if (i != 0)
            asmString += ", ";
        asmString += "$" + std::to_string(i);
    }
    asmString += "}, {$" + std::to_string(accumulatorSize) + ", $" +
                 std::to_string(accumulatorSize + 1) + ", $" +
                 std::to_string(accumulatorSize + 2) + ", $" +
                 std::to_string(accumulatorSize + 3) + "}, $" +
                 std::to_string(accumulatorSize + 4) + ", " +
                 std::to_string(*getConstantIntValue(args[3])) +
                 ", 1, 1, 1;\n}";

    std::string constraintString;
    for (size_t i = 0; i < constraints.size(); ++i) {
        if (i != 0)
            constraintString += ",";
        constraintString += constraints[i];
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, operands, asmString, constraintString,
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});

    auto resultType = mlir::cast<mlir::VectorType>(args[2].getType());
    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value result =
        mlir::vector::BroadcastOp::create(builder, location, resultType, zero);
    for (int64_t i = 0; i < accumulatorSize; ++i) {
        auto element = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{i});
        result = mlir::vector::InsertOp::create(builder, location, element,
                                                result, i);
    }
    return result;
}

mlir::Value NVVMIntrinsic::CreateWgmmaM64N192K32F32E4M3E4M3Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    constexpr int64_t kAccumulatorSize = 96;
    constexpr llvm::StringLiteral kName =
        "wgmma_m64n192k32_f32_e4m3_e4m3";
    if (!CheckRawWgmmaFunction(call_expr, ctx, resolved_args,
                               kAccumulatorSize, kName)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Type> elementTypes(kAccumulatorSize,
                                               builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    mlir::Value accumulatorStruct =
        mlir::LLVM::PoisonOp::create(builder, location, structType);
    for (int64_t i = 0; i < kAccumulatorSize; ++i) {
        auto element = mlir::vector::ExtractOp::create(
            builder, location, resolved_args[2], i);
        accumulatorStruct = mlir::LLVM::InsertValueOp::create(
            builder, location, structType, accumulatorStruct, element,
            llvm::ArrayRef<int64_t>{i});
    }

    auto scaleD = *getConstantIntValue(resolved_args[3]) == 0
                      ? mlir::NVVM::WGMMAScaleOut::zero
                      : mlir::NVVM::WGMMAScaleOut::one;
    auto result = mlir::NVVM::WgmmaMmaAsyncOp::create(
        builder, location, structType, accumulatorStruct, resolved_args[0],
        resolved_args[1],
        mlir::NVVM::MMAShapeAttr::get(builder.getContext(), 64, 192, 32),
        mlir::NVVM::WGMMATypes::e4m3, mlir::NVVM::WGMMATypes::e4m3,
        mlir::NVVM::WGMMATypes::f32, scaleD,
        mlir::NVVM::WGMMAScaleIn::one, mlir::NVVM::WGMMAScaleIn::one,
        mlir::NVVM::MMALayout::row, mlir::NVVM::MMALayout::col,
        mlir::NVVM::MMAIntOverflowAttr());

    auto resultType = mlir::cast<mlir::VectorType>(resolved_args[2].getType());
    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value resultVector = mlir::vector::BroadcastOp::create(
        builder, location, resultType, zero);
    for (int64_t i = 0; i < kAccumulatorSize; ++i) {
        auto element = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), result,
            llvm::ArrayRef<int64_t>{i});
        resultVector = mlir::vector::InsertOp::create(
            builder, location, element, resultVector, i);
    }
    return resultVector;
}

mlir::Value NVVMIntrinsic::CreateFloatToFP8Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool packed) const {
    if (!CheckFloatToFP8Function(call_expr, ctx, resolved_args, packed)) {
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto high =
        packed
            ? resolved_args[1]
            : mlir::arith::ConstantFloatOp::create(
                  builder, location, builder.getF32Type(), llvm::APFloat(0.0f))
                  .getResult();
    auto converted = mlir::NVVM::ConvertF32x2ToF8x2Op::create(
        builder, location, builder.getI16Type(), high, resolved_args[0],
        mlir::NVVM::FPRoundingMode::RN,
        mlir::NVVM::SaturationMode::SATFINITE,
        /*relu=*/false, mlir::Float8E4M3FNType::get(builder.getContext()));

    mlir::Value result = converted.getResult();
    if (!packed) {
        result = mlir::arith::TruncIOp::create(builder, location,
                                               builder.getI8Type(), result);
    }
    SetTypeInfo(result, TypeInfo{true});
    return result;
}

mlir::Value NVVMIntrinsic::CreateFloatX2ToBF16X2Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto report = [&]() -> mlir::Value {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "floatx2_to_bf16x2 requires exactly two f32 operands";
        return nullptr;
    };
    if (resolved_args.size() != 2 || !resolved_args[0] || !resolved_args[1] ||
        !resolved_args[0].getType().isF32() ||
        !resolved_args[1].getType().isF32()) {
        return report();
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto bf16x2 = mlir::VectorType::get({2}, builder.getBF16Type());
    auto converted = mlir::NVVM::ConvertF32x2ToBF16x2Op::create(
        builder, location, bf16x2,
        /*src_hi=*/resolved_args[1], /*src_lo=*/resolved_args[0],
        /*random_bits=*/mlir::Value{}, mlir::NVVM::FPRoundingMode::RN,
        mlir::NVVM::SaturationMode::NONE, /*relu=*/false);
    auto i32x1 = mlir::VectorType::get({1}, builder.getI32Type());
    auto bits = mlir::vector::BitCastOp::create(builder, location, i32x1,
                                                converted.getResult());
    auto result = mlir::vector::ExtractOp::create(builder, location, bits, 0);
    SetTypeInfo(result, TypeInfo{true});
    return result;
}

mlir::Value NVVMIntrinsic::CreateBF16X2ToFloatX2Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto report = [&]() -> mlir::Value {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "bf16x2_to_floatx2 requires exactly one i32 operand";
        return nullptr;
    };
    if (resolved_args.size() != 1 || !resolved_args[0] ||
        !resolved_args[0].getType().isInteger(32)) {
        return report();
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto i32x1 = mlir::VectorType::get({1}, builder.getI32Type());
    auto packed = mlir::vector::FromElementsOp::create(
        builder, location, i32x1, resolved_args);
    auto bf16x2 = mlir::VectorType::get({2}, builder.getBF16Type());
    auto values = mlir::vector::BitCastOp::create(builder, location, bf16x2,
                                                  packed);
    llvm::SmallVector<mlir::Value> converted;
    for (int64_t i = 0; i < 2; ++i) {
        auto value = mlir::vector::ExtractOp::create(builder, location, values,
                                                      i);
        converted.push_back(mlir::arith::ExtFOp::create(
            builder, location, builder.getF32Type(), value,
            mlir::arith::FastMathFlagsAttr{}));
    }
    return mlir::vector::FromElementsOp::create(
        builder, location,
        mlir::VectorType::get({2}, builder.getF32Type()), converted);
}

mlir::Value NVVMIntrinsic::CreatePackedBF16MathFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, size_t operandCount,
    llvm::StringRef instruction) const {
    if (resolved_args.size() != operandCount ||
        llvm::any_of(resolved_args, [](mlir::Value value) {
            return !value || !value.getType().isInteger(32);
        })) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "packed BF16 math requires " << operandCount
            << " i32 operands";
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    std::string constraints = "=r";
    for (size_t index = 0; index < operandCount; ++index) {
        constraints += ",r";
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, builder.getUnknownLoc(), builder.getI32Type(), resolved_args,
        instruction, constraints, /*hasSideEffects=*/false,
        /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    SetTypeInfo(inlineAsm.getRes(), TypeInfo{true});
    return inlineAsm.getRes();
}

mlir::Value NVVMIntrinsic::CreateF32MathFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool binary,
    llvm::StringRef instruction) const {
    size_t expected = binary ? 2 : 1;
    for (auto value : resolved_args) {
        if (!value || !value.getType().isF32()) {
            expected = 0;
            break;
        }
    }
    if (resolved_args.size() != expected) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << (binary ? "fmax requires two f32 operands"
                       : "fast_exp2 requires one f32 operand");
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto constraints = binary ? "=f,f,f" : "=f,f";
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, builder.getUnknownLoc(), builder.getF32Type(), resolved_args,
        instruction, constraints, /*hasSideEffects=*/false,
        /*isAlignStack=*/false, mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return inlineAsm.getRes();
}

mlir::Value NVVMIntrinsic::CreateF32FmaFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool fast) const {
    if (resolved_args.size() != 3 ||
        llvm::any_of(resolved_args, [](mlir::Value value) {
            return !value || !value.getType().isF32();
        })) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "fma requires three f32 operands";
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    llvm::SmallVector<mlir::Value> operands(resolved_args.begin(),
                                            resolved_args.end());
    bool negateMultiplicand = false;
    if (auto negate = operands[0].getDefiningOp<mlir::arith::NegFOp>()) {
        operands[0] = negate.getOperand();
        negateMultiplicand = true;
    }
    llvm::StringRef instruction;
    if (fast) {
        instruction = negateMultiplicand
                          ? "{ .reg .f32 neg_a; neg.f32 neg_a, $1; "
                            "fma.rn.ftz.f32 $0, neg_a, $2, $3; }"
                          : "fma.rn.ftz.f32 $0, $1, $2, $3;";
    } else {
        instruction = negateMultiplicand
                          ? "{ .reg .f32 neg_a; neg.f32 neg_a, $1; "
                            "fma.rn.f32 $0, neg_a, $2, $3; }"
                          : "fma.rn.f32 $0, $1, $2, $3;";
    }
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, builder.getUnknownLoc(), builder.getF32Type(), operands,
        instruction,
        "=f,f,f,f",
        /*hasSideEffects=*/false, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return inlineAsm.getRes();
}

bool NVVMIntrinsic::CheckFloatToFP8Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool packed) const {
    llvm::StringRef name = packed ? "floatx2_to_fp8x2" : "float_to_fp8";
    size_t expected = packed ? 2 : 1;
    if (resolved_args.size() != expected) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << name << " requires exactly " << expected << " f32 argument"
            << (packed ? "s" : "");
        return false;
    }
    for (auto value : resolved_args) {
        if (!value || !value.getType().isF32()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << name << " operands must be f32";
            return false;
        }
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaAsyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaAsyncFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto async = cf::NVVMWGMMAAsyncOp::create(
        builder, location, resolved_args[2].getType(), resolved_args[0],
        resolved_args[1], resolved_args[2]);
    return async.getResult();
}

bool NVVMIntrinsic::CheckWgmmaAsyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async requires exactly 3 arguments: desc_a, desc_b, acc";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for wgmma_async";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::WarpgroupMatrixDescriptorType>(
            resolved_args[0].getType()) ||
        !mlir::isa<mlir::nvgpu::WarpgroupMatrixDescriptorType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async descriptor operands must be wgmma_descriptor";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::WarpgroupAccumulatorType>(
            resolved_args[2].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async acc operand must be of type warpgroup_accumulator";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaInitAccumulatorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaInitAccumulatorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto mSize = *ConstantFolder::FoldIntValue(resolved_args[0]);
    auto nSize = *ConstantFolder::FoldIntValue(resolved_args[1]);
    auto vecType = mlir::VectorType::get({mSize, nSize}, builder.getF32Type());
    auto accType = mlir::nvgpu::WarpgroupAccumulatorType::get(
        builder.getContext(), vecType);

    auto acc = mlir::nvgpu::WarpgroupMmaInitAccumulatorOp::create(
        builder, location, accType);
    return acc.getResult();
}

bool NVVMIntrinsic::CheckWgmmaInitAccumulatorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_init_accumulator requires exactly 2 arguments: m, n";
        return false;
    }

    for (size_t i = 0; i < resolved_args.size(); ++i) {
        llvm::StringRef name = i == 0 ? "m" : "n";
        if (!resolved_args[i]) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name
                << " operand for wgmma_init_accumulator";
            return false;
        }
        if (!resolved_args[i].getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "wgmma_init_accumulator " << name
                << " operand must be an integer type";
            return false;
        }
        if (!ConstantFolder::FoldIntValue(resolved_args[i])) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "wgmma_init_accumulator " << name
                << " operand must be a constant integer";
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaStoreFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    cf::NVVMWGMMAStoreOp::create(builder, location, resolved_args[0],
                                 resolved_args[1]);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store requires exactly 2 arguments: acc, dst";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for wgmma_store";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::WarpgroupAccumulatorType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store acc operand must be of type warpgroup_accumulator";
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store dst operand must be of memref type";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierCreateFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierCreateFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    unsigned numBarriers = 1;
    if (!resolved_args.empty()) {
        numBarriers = static_cast<unsigned>(
            *ConstantFolder::FoldIntValue(resolved_args[0]));
    }
    auto workgroupSpace = mlir::gpu::AddressSpaceAttr::get(
        builder.getContext(), mlir::gpu::AddressSpace::Workgroup);
    auto barrierType = mlir::nvgpu::MBarrierGroupType::get(
        builder.getContext(), workgroupSpace, numBarriers);
    auto barrier =
        mlir::nvgpu::MBarrierCreateOp::create(builder, location, barrierType);
    return barrier.getResult();
}

bool NVVMIntrinsic::CheckMBarrierCreateFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() > 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_create takes at most one num_barriers argument";
        return false;
    }
    if (!resolved_args.empty()) {
        auto numBarriers = ConstantFolder::FoldIntValue(resolved_args[0]);
        if (!numBarriers || *numBarriers <= 0) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "mbarrier_create num_barriers must be a positive constant "
                   "integer";
            return false;
        }
    }

    return true;
}

static mlir::Value castMBarrierPredicate(mlir::OpBuilder &builder,
                                         mlir::Location location,
                                         mlir::Value value) {
    if (!value) {
        return {};
    }
    return castIntegerTo(builder, location, value, builder.getI1Type());
}

static bool checkMBarrierIntegerOperand(ast::Call *call_expr,
                                        GeneratorContext *ctx,
                                        mlir::Value value,
                                        llvm::StringRef operation,
                                        llvm::StringRef operand) {
    if (!value || !value.getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << operation << " " << operand
            << " operand must be an integer type";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierInitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierInitFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value mbarId = positionalCount > 1 ? resolved_args[1] : mlir::Value();
    mlir::Value count =
        positionalCount > 2
            ? resolved_args[2]
            : mlir::arith::ConstantIndexOp::create(builder, location, 0)
                  .getResult();
    mlir::Value predicate =
        positionalCount > 3 ? resolved_args[3] : mlir::Value();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "mbar_id") {
            mbarId = value;
        } else if (name == "count") {
            count = value;
        } else if (name == "predicate") {
            predicate = value;
        }
    }

    mbarId = castToIndex(builder, location, mbarId);
    count = castToIndex(builder, location, count);
    if (predicate) {
        predicate = castMBarrierPredicate(builder, location, predicate);
    }
    if (!mbarId || !count || (predicate && !predicate.getType().isInteger(1))) {
        return nullptr;
    }
    mlir::nvgpu::MBarrierInitOp::create(builder, location, resolved_args[0],
                                        count, mbarId, predicate);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierInitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 1 || positionalCount > 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init requires barrier, mbar_id and optional count, "
               "predicate";
        return false;
    }

    for (auto name : keywords) {
        if (name != "mbar_id" && name != "count" && name != "predicate") {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "mbarrier_init got unsupported keyword argument '" << name
                << "'";
            return false;
        }
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate barrier operand for mbarrier_init";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init operand must be of type mbarrier_group_t";
        return false;
    }

    auto keywordValue = [&](llvm::StringRef wanted) -> mlir::Value {
        for (size_t i = 0; i < keywordCount; ++i) {
            if (keywords[i] == wanted) {
                return resolved_args[positionalCount + i];
            }
        }
        return {};
    };

    mlir::Value mbarId =
        positionalCount > 1 ? resolved_args[1] : keywordValue("mbar_id");
    if (!mbarId) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate mbar_id operand for mbarrier_init";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "mbarrier_init " << name
                << " operand must be an integer type";
            return false;
        }
        return true;
    };

    if (!checkIntArg(mbarId, "mbar_id")) {
        return false;
    }
    if (positionalCount > 2 && !checkIntArg(resolved_args[2], "count")) {
        return false;
    }
    if (positionalCount > 3 && !checkIntArg(resolved_args[3], "predicate")) {
        return false;
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        if (!checkIntArg(resolved_args[positionalCount + i], keywords[i])) {
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierTryWaitParityFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierTryWaitParityFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto parity = castMBarrierPredicate(builder, location, resolved_args[1]);
    auto ticks = castToIndex(builder, location, resolved_args[2]);
    auto mbarId = castToIndex(builder, location, resolved_args[3]);
    if (!parity || !ticks || !mbarId) {
        return nullptr;
    }
    mlir::nvgpu::MBarrierTryWaitParityOp::create(
        builder, location, resolved_args[0], parity, ticks, mbarId);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierTryWaitParityFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity requires exactly 4 arguments: "
               "barrier, phaseParity, ticks, mbarId";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_try_wait_parity";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity operand must be of type "
               "mbarrier_group_t";
        return false;
    }

    for (size_t i = 1; i < resolved_args.size(); ++i) {
        static constexpr llvm::StringLiteral names[] = {"phaseParity", "ticks",
                                                         "mbarId"};
        if (!checkMBarrierIntegerOperand(call_expr, ctx, resolved_args[i],
                                         "mbarrier_try_wait_parity",
                                         names[i - 1])) {
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierArriveFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierArriveFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto mbarId = castToIndex(builder, location, resolved_args[1]);
    if (!mbarId) {
        return nullptr;
    }
    auto token = mlir::nvgpu::MBarrierArriveOp::create(
        builder, location, resolved_args[0], mbarId);
    return token.getResult();
}

bool NVVMIntrinsic::CheckMBarrierArriveFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive requires exactly 2 arguments: barrier, mbarId";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_arrive";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive operand must be of type mbarrier_group_t";
        return false;
    }

    if (!checkMBarrierIntegerOperand(call_expr, ctx, resolved_args[1],
                                     "mbarrier_arrive", "mbarId")) {
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierArriveClusterFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierArriveClusterFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto mbarId = castToIndex(builder, location, resolved_args[1]);
    auto ctaId = castIntegerTo(builder, location, resolved_args[2],
                               builder.getI32Type());
    if (!mbarId || !ctaId) {
        return nullptr;
    }

    // Retrieve the shared-memory address of the selected barrier before
    // mapping it into the target CTA's cluster shared-memory address space.
    auto sharedAddress = mlir::nvgpu::MBarrierGetOp::create(
        builder, location, builder.getI32Type(), resolved_args[0], mbarId);
    auto sharedPointerType = mlir::LLVM::LLVMPointerType::get(
        builder.getContext(), static_cast<unsigned>(
                                  mlir::NVVM::NVVMMemorySpace::Shared));
    auto sharedPointer = mlir::LLVM::IntToPtrOp::create(
        builder, location, sharedPointerType, sharedAddress.getResult());
    auto clusterPointerType = mlir::LLVM::LLVMPointerType::get(
        builder.getContext(), static_cast<unsigned>(
                                  mlir::NVVM::NVVMMemorySpace::SharedCluster));
    auto remotePointer = mlir::NVVM::MapaOp::create(
        builder, location, clusterPointerType, sharedPointer.getResult(),
        ctaId);
    mlir::NVVM::MBarrierArriveOp::create(
        builder, location, mlir::Type(), remotePointer.getResult(),
        mlir::Value(), mlir::NVVM::MemScopeKind::CTA);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierArriveClusterFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_cluster requires exactly 3 arguments: "
               "barrier, mbarId, ctaId";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_arrive_cluster";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_cluster operand must be of type "
               "mbarrier_group_t";
        return false;
    }

    return checkMBarrierIntegerOperand(call_expr, ctx, resolved_args[1],
                                       "mbarrier_arrive_cluster", "mbarId") &&
           checkMBarrierIntegerOperand(call_expr, ctx, resolved_args[2],
                                       "mbarrier_arrive_cluster", "ctaId");
}

mlir::Value NVVMIntrinsic::CreateMBarrierTestWaitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierTestWaitFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto mbarId = castToIndex(builder, location, resolved_args[2]);
    if (!mbarId) {
        return nullptr;
    }
    auto ready = mlir::nvgpu::MBarrierTestWaitOp::create(
        builder, location, resolved_args[0], resolved_args[1], mbarId);
    return ready.getResult();
}

bool NVVMIntrinsic::CheckMBarrierTestWaitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait requires exactly 3 arguments: barrier, "
               "token, mbarId";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_test_wait";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait operand must be of type mbarrier_group_t";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierTokenType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait token operand must be of type "
               "mbarrier_token_t";
        return false;
    }

    if (!checkMBarrierIntegerOperand(call_expr, ctx, resolved_args[2],
                                     "mbarrier_test_wait", "mbarId")) {
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierArriveExpectTxFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierArriveExpectTxFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto txCount = castToIndex(builder, location, resolved_args[1]);
    auto mbarId = castToIndex(builder, location, resolved_args[2]);
    auto predicate =
        castMBarrierPredicate(builder, location, resolved_args[3]);
    if (!txCount || !mbarId || !predicate) {
        return nullptr;
    }
    mlir::nvgpu::MBarrierArriveExpectTxOp::create(
        builder, location, resolved_args[0], txCount, mbarId, predicate);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierArriveExpectTxFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx requires exactly 4 arguments: "
               "barrier, txcount, mbarId, predicate";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_arrive_expect_tx";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx operand must be of type "
               "mbarrier_group_t";
        return false;
    }

    for (size_t i = 1; i < resolved_args.size(); ++i) {
        static constexpr llvm::StringLiteral names[] = {"txcount", "mbarId",
                                                         "predicate"};
        if (!checkMBarrierIntegerOperand(call_expr, ctx, resolved_args[i],
                                         "mbarrier_arrive_expect_tx",
                                         names[i - 1])) {
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncSharedGlobalFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool cacheAll) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncCaSharedGlobalFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto toIndex = [&](mlir::Value value) -> mlir::Value {
        if (value.getType().isIndex()) {
            return value;
        }
        return mlir::arith::IndexCastOp::create(builder, location,
                                                builder.getIndexType(), value);
    };

    auto dstBase = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), resolved_args[0]);
    auto srcBase = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), resolved_args[1]);
    auto dstOffset = toIndex(resolved_args[2]);
    auto srcOffset = toIndex(resolved_args[3]);
    auto dstAddr = mlir::arith::AddIOp::create(builder, location,
                                               dstBase.getResult(), dstOffset);
    auto srcAddr = mlir::arith::AddIOp::create(builder, location,
                                               srcBase.getResult(), srcOffset);

    auto dstAddrI32 = mlir::arith::IndexCastOp::create(
        builder, location, builder.getI32Type(), dstAddr.getResult());
    auto srcAddrI32 = mlir::arith::IndexCastOp::create(
        builder, location, builder.getI32Type(), srcAddr.getResult());
    auto dstAddrI64 = mlir::arith::ExtUIOp::create(
        builder, location, builder.getI64Type(), dstAddrI32.getResult());
    auto srcAddrI64 = mlir::arith::ExtUIOp::create(
        builder, location, builder.getI64Type(), srcAddrI32.getResult());

    auto dstPtrType = mlir::LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Shared));
    auto srcPtrType = mlir::LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Global));

    auto dstPtr = mlir::LLVM::IntToPtrOp::create(builder, location, dstPtrType,
                                                 dstAddrI64.getResult());
    auto srcPtr = mlir::LLVM::IntToPtrOp::create(builder, location, srcPtrType,
                                                 srcAddrI64.getResult());
    auto sizeBytes =
        static_cast<int64_t>(*ConstantFolder::FoldIntValue(resolved_args[4]));

    mlir::NVVM::CpAsyncOp::create(
        builder, location, dstPtr.getResult(), srcPtr.getResult(),
        builder.getI32IntegerAttr(sizeBytes),
        mlir::NVVM::LoadCacheModifierKindAttr::get(
            builder.getContext(),
            cacheAll ? mlir::NVVM::LoadCacheModifierKind::CA
                     : mlir::NVVM::LoadCacheModifierKind::CG),
        mlir::Value());

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncCaSharedGlobalFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 5) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global requires 5 arguments: dst, src, "
               "dst_offset_bytes, src_offset_bytes, size_bytes";
        return false;
    }
    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3] || !resolved_args[4]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "failed to generate one or more operands for "
               "cp_async_ca_shared_global";
        return false;
    }

    auto dstType = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    auto srcType = mlir::dyn_cast<cf::MemRefType>(resolved_args[1].getType());
    if (!dstType || !srcType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global dst/src operands must be memrefs";
        return false;
    }

    auto gpuSpace = mlir::gpu::AddressSpaceAttr::get(
        ctx->GetCurrentFunctionGenerator()->GetBuilder().getContext(),
        mlir::gpu::AddressSpace::Workgroup);
    if (dstType.getMemorySpace() != gpuSpace) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global dst operand must be in workgroup "
               "memory";
        return false;
    }

    if (!resolved_args[2].getType().isIntOrIndex() ||
        !resolved_args[3].getType().isIntOrIndex() ||
        !resolved_args[4].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global offset/size operands must be "
               "integer/index";
        return false;
    }

    auto sizeBytes = ConstantFolder::FoldIntValue(resolved_args[4]);
    if (!sizeBytes) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global size_bytes must be a compile-time "
               "constant";
        return false;
    }

    if (*sizeBytes != 4 && *sizeBytes != 8 && *sizeBytes != 16) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global size_bytes must be one of 4, 8, or "
               "16";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncCommitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncCommitGroupFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    mlir::NVVM::CpAsyncCommitGroupOp::create(builder, location);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncCommitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_commit_group takes no arguments";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncWaitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncWaitGroupFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto group = static_cast<int32_t>(
        *ConstantFolder::FoldIntValue(resolved_args[0]));
    mlir::NVVM::CpAsyncWaitGroupOp::create(builder, location,
                                           builder.getI32IntegerAttr(group));
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncWaitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1 || !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group requires 1 argument: n";
        return false;
    }
    if (!resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n operand must be integer/index";
        return false;
    }

    auto group = ConstantFolder::FoldIntValue(resolved_args[0]);
    if (!group) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n must be a compile-time constant";
        return false;
    }
    if (*group < 0 || *group > 8) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n must be in [0, 8]";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncBulkFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    CpAsyncBulkIntrinsicKind kind) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncBulkFunction(call_expr, ctx, resolved_args, kind)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    auto keywordValue = [&](llvm::StringRef key) -> mlir::Value {
        for (size_t i = 0; i < keywordCount; ++i) {
            if (llvm::StringRef(keywords[i]) == key) {
                return resolved_args[positionalCount + i];
            }
        }
        return {};
    };
    auto optionalValue = [&](size_t positionalIndex,
                             llvm::StringRef key) -> mlir::Value {
        if (positionalCount > positionalIndex) {
            return resolved_args[positionalIndex];
        }
        return keywordValue(key);
    };
    auto offsetValue = [&](size_t positionalIndex,
                           llvm::StringRef key) -> mlir::Value {
        if (auto value = optionalValue(positionalIndex, key)) {
            return value;
        }
        return createDefaultIndex(builder, location);
    };
    auto i32 = [&](mlir::Value value) {
        return castIntegerTo(builder, location, value, builder.getI32Type());
    };
    auto i16 = [&](mlir::Value value) {
        return castIntegerTo(builder, location, value, builder.getI16Type());
    };
    auto i64 = [&](mlir::Value value) {
        return castIntegerTo(builder, location, value, builder.getI64Type());
    };
    auto i1 = [&](mlir::Value value) {
        return castIntegerTo(builder, location, value, builder.getI1Type());
    };
    auto coordinates = [&](mlir::Value tuple) {
        llvm::SmallVector<mlir::Value> values;
        extractTupleValues(tuple, values);
        for (auto &value : values) {
            value = i32(value);
        }
        return values;
    };
    auto im2colOffsets = [&](mlir::Value tuple) {
        llvm::SmallVector<mlir::Value> values;
        if (tuple) {
            extractTupleValues(tuple, values);
        }
        for (auto &value : values) {
            value = i16(value);
        }
        return values;
    };

    switch (kind) {
    case CpAsyncBulkIntrinsicKind::CommitGroup:
        mlir::NVVM::CpAsyncBulkCommitGroupOp::create(builder, location);
        break;
    case CpAsyncBulkIntrinsicKind::WaitGroup: {
        auto group =
            static_cast<uint32_t>(
                *ConstantFolder::FoldIntValue(resolved_args[0]));
        bool read = false;
        if (auto readValue = optionalValue(1, "read")) {
            read = *ConstantFolder::FoldIntValue(readValue) != 0;
        }
        mlir::NVVM::CpAsyncBulkWaitGroupOp::create(builder, location, group,
                                                   read ? builder.getUnitAttr()
                                                        : mlir::UnitAttr{});
        break;
    }
    case CpAsyncBulkIntrinsicKind::Prefetch: {
        auto src = createPointerFromMemRef(builder, location, resolved_args[0],
                                           offsetValue(2, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Global);
        mlir::NVVM::CpAsyncBulkPrefetchOp::create(
            builder, location, src, i32(resolved_args[1]),
            i64(keywordValue("l2_cache_hint")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::GlobalSharedCTA: {
        auto dst = createPointerFromMemRef(builder, location, resolved_args[0],
                                           offsetValue(3, "dst_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Global);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(4, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Shared);
        mlir::NVVM::CpAsyncBulkSharedCTAToGlobalOp::create(
            builder, location, dst, src, i32(resolved_args[2]),
            i64(keywordValue("l2_cache_hint")), i16(keywordValue("byte_mask")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::SharedClusterGlobal: {
        auto dst =
            createPointerFromMemRef(builder, location, resolved_args[0],
                                    offsetValue(4, "dst_offset_bytes"),
                                    mlir::NVVM::NVVMMemorySpace::SharedCluster);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(5, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Global);
        auto mbar =
            createSharedBarrierPointer(builder, location, resolved_args[2],
                                       offsetValue(6, "mbar_offset_bytes"));
        mlir::NVVM::CpAsyncBulkGlobalToSharedClusterOp::create(
            builder, location, dst, src, mbar, i32(resolved_args[3]),
            i16(keywordValue("multicast_mask")),
            i64(keywordValue("l2_cache_hint")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::SharedClusterSharedCTA: {
        auto dst =
            createPointerFromMemRef(builder, location, resolved_args[0],
                                    offsetValue(4, "dst_offset_bytes"),
                                    mlir::NVVM::NVVMMemorySpace::SharedCluster);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(5, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Shared);
        auto mbar =
            createSharedBarrierPointer(builder, location, resolved_args[2],
                                       offsetValue(6, "mbar_offset_bytes"));
        mlir::NVVM::CpAsyncBulkSharedCTAToSharedClusterOp::create(
            builder, location, dst, src, mbar, i32(resolved_args[3]));
        break;
    }
    case CpAsyncBulkIntrinsicKind::TensorGlobalSharedCTA: {
        auto desc =
            createDescriptorPointer(builder, location, resolved_args[0]);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(3, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Shared);
        auto coords = coordinates(resolved_args[2]);
        mlir::Value predicate;
        if (auto value = keywordValue("predicate")) {
            predicate = i1(value);
        }
        mlir::NVVM::CpAsyncBulkTensorSharedCTAToGlobalOp::create(
            builder, location, desc, src, coords,
            i64(keywordValue("l2_cache_hint")), mlir::NVVM::TMAStoreMode::TILE,
            predicate);
        break;
    }
    case CpAsyncBulkIntrinsicKind::TensorPrefetch: {
        auto desc =
            createDescriptorPointer(builder, location, resolved_args[0]);
        auto coords = coordinates(resolved_args[1]);
        auto im2col = im2colOffsets(keywordValue("im2col_offsets"));
        auto mode = im2col.empty() ? mlir::NVVM::TMALoadMode::TILE
                                   : mlir::NVVM::TMALoadMode::IM2COL;
        mlir::NVVM::CpAsyncBulkTensorPrefetchOp::create(
            builder, location, desc, coords, im2col, mode,
            i64(keywordValue("l2_cache_hint")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::TensorReduce: {
        auto desc =
            createDescriptorPointer(builder, location, resolved_args[0]);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(4, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Shared);
        auto coords = coordinates(resolved_args[2]);
        auto redKindValue =
            static_cast<uint32_t>(
                *ConstantFolder::FoldIntValue(resolved_args[3]));
        auto redKind = *mlir::NVVM::symbolizeTMAReduxKind(redKindValue);
        mlir::NVVM::CpAsyncBulkTensorReduceOp::create(
            builder, location, desc, src, redKind,
            mlir::NVVM::TMAStoreMode::TILE, coords,
            i64(keywordValue("l2_cache_hint")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::TensorSharedClusterGlobal: {
        auto dst =
            createPointerFromMemRef(builder, location, resolved_args[0],
                                    offsetValue(4, "dst_offset_bytes"),
                                    mlir::NVVM::NVVMMemorySpace::SharedCluster);
        auto desc =
            createDescriptorPointer(builder, location, resolved_args[1]);
        auto coords = coordinates(resolved_args[2]);
        auto mbar =
            createSharedBarrierPointer(builder, location, resolved_args[3],
                                       offsetValue(5, "mbar_offset_bytes"));
        auto im2col = im2colOffsets(keywordValue("im2col_offsets"));
        auto mode = im2col.empty() ? mlir::NVVM::TMALoadMode::TILE
                                   : mlir::NVVM::TMALoadMode::IM2COL;
        mlir::Value predicate;
        if (auto value = keywordValue("predicate")) {
            predicate = i1(value);
        }
        mlir::NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp::create(
            builder, location, dst, desc, coords, mbar, im2col,
            i16(keywordValue("multicast_mask")),
            i64(keywordValue("l2_cache_hint")), mode,
            /*isCTAOnly=*/false, mlir::NVVM::CTAGroupKindAttr{}, predicate);
        break;
    }
    }

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncBulkFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    CpAsyncBulkIntrinsicKind kind) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_bulk internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    auto fail = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
        return false;
    };
    auto checkPositionalCount = [&](size_t min, size_t max,
                                    llvm::StringRef message) {
        if (positionalCount < min || positionalCount > max) {
            return fail(message);
        }
        return true;
    };
    auto keywordAllowed = [&](llvm::StringRef keyword,
                              std::initializer_list<llvm::StringRef> allowed) {
        for (auto candidate : allowed) {
            if (keyword == candidate) {
                return true;
            }
        }
        return false;
    };
    auto checkKeywords =
        [&](std::initializer_list<llvm::StringRef> allowed) -> bool {
        for (auto name : keywords) {
            if (!keywordAllowed(name, allowed)) {
                std::string message =
                    "unsupported cp_async_bulk keyword argument '";
                message += llvm::StringRef(name).str();
                message += "'";
                return fail(message);
            }
        }
        return true;
    };
    auto checkPresent = [&](size_t index, llvm::StringRef name) {
        if (index >= resolved_args.size() || !resolved_args[index]) {
            std::string message = "failed to generate cp_async_bulk operand '";
            message += name.str();
            message += "'";
            return fail(message);
        }
        return true;
    };
    auto checkInt = [&](mlir::Value value, llvm::StringRef name) {
        if (value && !value.getType().isIntOrIndex()) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be an integer/index";
            return fail(message);
        }
        return true;
    };
    auto checkMemRef = [&](mlir::Value value, llvm::StringRef name) {
        if (!value ||
            (!isMemRefLike(value.getType()) &&
             !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be a memref-like value";
            return fail(message);
        }
        return true;
    };
    auto checkDescriptor = [&](mlir::Value value, llvm::StringRef name) {
        if (!value ||
            (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
                 value.getType()) &&
             !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be a TMA descriptor";
            return fail(message);
        }
        return true;
    };
    auto checkBarrier = [&](mlir::Value value, llvm::StringRef name) {
        if (!value ||
            (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(value.getType()) &&
             !isMemRefLike(value.getType()) &&
             !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be an mbarrier or shared pointer";
            return fail(message);
        }
        return true;
    };
    auto checkTuple = [&](mlir::Value value, llvm::StringRef name) {
        llvm::SmallVector<mlir::Value> values;
        if (!value || !extractTupleValues(value, values) || values.empty()) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be a non-empty coordinate tuple";
            return fail(message);
        }
        for (auto elem : values) {
            if (!checkInt(elem, name)) {
                return false;
            }
        }
        return true;
    };
    auto checkOptionalKeywordInts =
        [&](std::initializer_list<llvm::StringRef> intNames) -> bool {
        for (size_t i = 0; i < keywordCount; ++i) {
            llvm::StringRef name = keywords[i];
            for (auto intName : intNames) {
                if (name == intName &&
                    !checkInt(resolved_args[positionalCount + i], name)) {
                    return false;
                }
            }
        }
        return true;
    };

    switch (kind) {
    case CpAsyncBulkIntrinsicKind::CommitGroup:
        return checkPositionalCount(
                   0, 0, "cp_async_bulk_commit_group takes no arguments") &&
               checkKeywords({});
    case CpAsyncBulkIntrinsicKind::WaitGroup: {
        if (!checkPositionalCount(
                1, 2,
                "cp_async_bulk_wait_group requires group and optional read") ||
            !checkKeywords({"read"}) || !checkPresent(0, "group") ||
            !checkInt(resolved_args[0], "group")) {
            return false;
        }
        auto group = ConstantFolder::FoldIntValue(resolved_args[0]);
        if (!group || *group < 0) {
            return fail("cp_async_bulk_wait_group group must be a non-negative "
                        "compile-time constant");
        }
        if (positionalCount > 1 && (!checkInt(resolved_args[1], "read") ||
                                    !ConstantFolder::FoldIntValue(
                                        resolved_args[1]))) {
            return fail("cp_async_bulk_wait_group read must be a compile-time "
                        "constant");
        }
        return checkOptionalKeywordInts({"read"});
    }
    case CpAsyncBulkIntrinsicKind::Prefetch:
        if (!checkPositionalCount(2, 3,
                                  "cp_async_bulk_prefetch requires src, size "
                                  "and optional src_offset_bytes") ||
            !checkKeywords({"src_offset_bytes", "l2_cache_hint"}) ||
            !checkMemRef(resolved_args[0], "src") ||
            !checkInt(resolved_args[1], "size")) {
            return false;
        }
        if (positionalCount > 2 &&
            !checkInt(resolved_args[2], "src_offset_bytes")) {
            return false;
        }
        return checkOptionalKeywordInts({"src_offset_bytes", "l2_cache_hint"});
    case CpAsyncBulkIntrinsicKind::GlobalSharedCTA:
        if (!checkPositionalCount(3, 5,
                                  "cp_async_bulk_global_shared_cta requires "
                                  "dst, src, size and optional offsets") ||
            !checkKeywords({"dst_offset_bytes", "src_offset_bytes",
                            "l2_cache_hint", "byte_mask"}) ||
            !checkMemRef(resolved_args[0], "dst") ||
            !checkMemRef(resolved_args[1], "src") ||
            !checkInt(resolved_args[2], "size")) {
            return false;
        }
        for (size_t i = 3; i < positionalCount; ++i) {
            if (!checkInt(resolved_args[i],
                          i == 3 ? "dst_offset_bytes" : "src_offset_bytes")) {
                return false;
            }
        }
        return checkOptionalKeywordInts({"dst_offset_bytes", "src_offset_bytes",
                                         "l2_cache_hint", "byte_mask"});
    case CpAsyncBulkIntrinsicKind::SharedClusterGlobal:
    case CpAsyncBulkIntrinsicKind::SharedClusterSharedCTA: {
        bool isClusterGlobal =
            kind == CpAsyncBulkIntrinsicKind::SharedClusterGlobal;
        if (!checkPositionalCount(
                4, 7,
                isClusterGlobal
                    ? "cp_async_bulk_shared_cluster_global requires dst, src, "
                      "mbar, size and optional offsets"
                    : "cp_async_bulk_shared_cluster_shared_cta requires dst, "
                      "src, mbar, size and optional offsets") ||
            !checkKeywords(isClusterGlobal
                               ? std::initializer_list<
                                     llvm::StringRef>{"dst_offset_bytes",
                                                      "src_offset_bytes",
                                                      "mbar_offset_bytes",
                                                      "multicast_mask",
                                                      "l2_cache_hint"}
                               : std::initializer_list<
                                     llvm::StringRef>{"dst_offset_bytes",
                                                      "src_offset_bytes",
                                                      "mbar_offset_bytes"}) ||
            !checkMemRef(resolved_args[0], "dst") ||
            !checkMemRef(resolved_args[1], "src") ||
            !checkBarrier(resolved_args[2], "mbar") ||
            !checkInt(resolved_args[3], "size")) {
            return false;
        }
        for (size_t i = 4; i < positionalCount; ++i) {
            if (!checkInt(resolved_args[i], "offset")) {
                return false;
            }
        }
        return checkOptionalKeywordInts({"dst_offset_bytes", "src_offset_bytes",
                                         "mbar_offset_bytes", "multicast_mask",
                                         "l2_cache_hint"});
    }
    case CpAsyncBulkIntrinsicKind::TensorGlobalSharedCTA:
        if (!checkPositionalCount(
                3, 4,
                "cp_async_bulk_tensor_global_shared_cta requires desc, src, "
                "coords and optional src_offset_bytes") ||
            !checkKeywords(
                {"src_offset_bytes", "l2_cache_hint", "predicate"}) ||
            !checkDescriptor(resolved_args[0], "desc") ||
            !checkMemRef(resolved_args[1], "src") ||
            !checkTuple(resolved_args[2], "coords")) {
            return false;
        }
        if (positionalCount > 3 &&
            !checkInt(resolved_args[3], "src_offset_bytes")) {
            return false;
        }
        return checkOptionalKeywordInts(
            {"src_offset_bytes", "l2_cache_hint", "predicate"});
    case CpAsyncBulkIntrinsicKind::TensorPrefetch:
        if (!checkPositionalCount(
                2, 2,
                "cp_async_bulk_tensor_prefetch requires desc and coords") ||
            !checkKeywords({"im2col_offsets", "l2_cache_hint"}) ||
            !checkDescriptor(resolved_args[0], "desc") ||
            !checkTuple(resolved_args[1], "coords")) {
            return false;
        }
        for (size_t i = 0; i < keywordCount; ++i) {
            if (llvm::StringRef(keywords[i]) == "im2col_offsets" &&
                resolved_args[positionalCount + i] &&
                !checkTuple(resolved_args[positionalCount + i],
                            "im2col_offsets")) {
                return false;
            }
        }
        return checkOptionalKeywordInts({"l2_cache_hint"});
    case CpAsyncBulkIntrinsicKind::TensorReduce: {
        if (!checkPositionalCount(
                4, 5,
                "cp_async_bulk_tensor_reduce requires desc, src, coords, "
                "red_kind and optional src_offset_bytes") ||
            !checkKeywords({"src_offset_bytes", "l2_cache_hint"}) ||
            !checkDescriptor(resolved_args[0], "desc") ||
            !checkMemRef(resolved_args[1], "src") ||
            !checkTuple(resolved_args[2], "coords") ||
            !checkInt(resolved_args[3], "red_kind")) {
            return false;
        }
        auto redKind = ConstantFolder::FoldIntValue(resolved_args[3]);
        if (!redKind || *redKind < 0 ||
            !mlir::NVVM::symbolizeTMAReduxKind(
                static_cast<uint32_t>(*redKind))) {
            return fail(
                "cp_async_bulk_tensor_reduce red_kind must be one of 0..7");
        }
        if (positionalCount > 4 &&
            !checkInt(resolved_args[4], "src_offset_bytes")) {
            return false;
        }
        return checkOptionalKeywordInts({"src_offset_bytes", "l2_cache_hint"});
    }
    case CpAsyncBulkIntrinsicKind::TensorSharedClusterGlobal:
        if (!checkPositionalCount(
                4, 6,
                "cp_async_bulk_tensor_shared_cluster_global requires dst, "
                "desc, coords, mbar and optional offsets") ||
            !checkKeywords({"dst_offset_bytes", "mbar_offset_bytes",
                            "im2col_offsets", "multicast_mask", "l2_cache_hint",
                            "predicate"}) ||
            !checkMemRef(resolved_args[0], "dst") ||
            !checkDescriptor(resolved_args[1], "desc") ||
            !checkTuple(resolved_args[2], "coords") ||
            !checkBarrier(resolved_args[3], "mbar")) {
            return false;
        }
        for (size_t i = 4; i < positionalCount; ++i) {
            if (!checkInt(resolved_args[i], "offset")) {
                return false;
            }
        }
        for (size_t i = 0; i < keywordCount; ++i) {
            if (llvm::StringRef(keywords[i]) == "im2col_offsets" &&
                resolved_args[positionalCount + i] &&
                !checkTuple(resolved_args[positionalCount + i],
                            "im2col_offsets")) {
                return false;
            }
        }
        return checkOptionalKeywordInts({"dst_offset_bytes",
                                         "mbar_offset_bytes", "multicast_mask",
                                         "l2_cache_hint", "predicate"});
    }

    llvm_unreachable("unknown cp.async.bulk intrinsic kind");
}

mlir::Value NVVMIntrinsic::CreateMakeTMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMakeTMADescriptorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value swizzleValue =
        mlir::arith::ConstantIndexOp::create(builder, location, 0).getResult();
    if (positionalCount > 2) {
        swizzleValue = resolved_args[2];
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        if (llvm::StringRef(keywords[i]) == "swizzle_kind") {
            swizzleValue = resolved_args[positionalCount + i];
        }
    }
    auto swizzleInt = ConstantFolder::FoldIntValue(swizzleValue);
    auto swizzleKind = getTensorMapSwizzleKind(*swizzleInt);

    auto tensorType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!tensorType) {
        return nullptr;
    }

    auto layoutOp = resolved_args[1].getDefiningOp<cf::MakeLayoutOp>();
    llvm::SmallVector<int64_t> smemDims;
    llvm::SmallVector<int64_t> smemStrides;
    if (!layoutOp ||
        !extractConstantTupleValues(layoutOp.getDims(), smemDims) ||
        !extractConstantTupleValues(layoutOp.getStride(), smemStrides) ||
        smemDims.size() != smemStrides.size()) {
        return nullptr;
    }

    auto workgroupMemorySpace = mlir::IntegerAttr::get(
        mlir::IntegerType::get(builder.getContext(), 64), 3);
    auto descriptorTensorType = mlir::MemRefType::get(
        smemDims, tensorType.getElementType(),
        mlir::StridedLayoutAttr::get(builder.getContext(), 0, smemStrides),
        workgroupMemorySpace);

    auto resultType = mlir::nvgpu::TensorMapDescriptorType::get(
        builder.getContext(), descriptorTensorType,
        *swizzleKind,
        mlir::nvgpu::TensorMapL2PromoKind::L2PROMO_NONE,
        mlir::nvgpu::TensorMapOOBKind::OOB_ZERO,
        mlir::nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);

    auto descriptor = cf::NVVMTMADescriptorOp::create(
        builder, location, resultType, resolved_args[0], resolved_args[1],
        swizzleValue);
    return descriptor.getResult();
}

bool NVVMIntrinsic::CheckMakeTMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 2 || positionalCount > 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires tensor, smem_layout and "
               "optional swizzle_kind";
        return false;
    }

    for (auto name : keywords) {
        if (llvm::StringRef(name) != "swizzle_kind") {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_tma_descriptor got unsupported keyword argument '"
                << name << "'";
            return false;
        }
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate tensor operand for make_tma_descriptor";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate smem_layout operand for make_tma_descriptor";
        return false;
    }

    auto tensorType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!tensorType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor expects a memref tensor operand";
        return false;
    }

    auto layoutOp = resolved_args[1].getDefiningOp<cf::MakeLayoutOp>();
    if (!layoutOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor expects smem_layout from make_layout()";
        return false;
    }

    auto dimsTuple = layoutOp.getDims().getDefiningOp<cf::MakeIntTupleOp>();
    if (!dimsTuple || dimsTuple.getNumElements() == 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires a non-empty smem_layout";
        return false;
    }

    llvm::SmallVector<int64_t> smemDims;
    llvm::SmallVector<int64_t> smemStrides;
    if (!extractConstantTupleValues(layoutOp.getDims(), smemDims) ||
        !extractConstantTupleValues(layoutOp.getStride(), smemStrides) ||
        smemDims.size() != smemStrides.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires a static smem_layout";
        return false;
    }
    if (smemDims.size() != static_cast<size_t>(tensorType.getRank())) {
        std::string message =
            "make_tma_descriptor smem_layout rank must match tensor rank; "
            "got layout rank " +
            std::to_string(smemDims.size()) + " and tensor rank " +
            std::to_string(tensorType.getRank());
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
        return false;
    }

    mlir::Value swizzleValue;
    if (positionalCount > 2) {
        swizzleValue = resolved_args[2];
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        if (llvm::StringRef(keywords[i]) == "swizzle_kind") {
            swizzleValue = resolved_args[positionalCount + i];
        }
    }
    if (swizzleValue) {
        if (!swizzleValue.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_tma_descriptor swizzle_kind must be an integer";
            return false;
        }
        auto swizzleInt = ConstantFolder::FoldIntValue(swizzleValue);
        if (!swizzleInt || !getTensorMapSwizzleKind(*swizzleInt)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_tma_descriptor swizzle_kind must be one of 0, 1, 2, "
                   "or 3";
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMAFenceFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMAFenceFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    cf::NVVMTMAFenceOp::create(builder, location,
                               mlir::ValueRange{resolved_args[0]},
                               mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMAFenceFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_fence requires exactly 1 argument: desc";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate desc operand for tma_fence";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_fence desc operand must be a TMA descriptor";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMAPrefetchDescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMAPrefetchDescriptorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto descriptor =
        createDescriptorPointer(builder, location, resolved_args[0]);
    mlir::NVVM::PrefetchOp::create(
        builder, location, mlir::NVVM::PrefetchCacheLevelAttr{},
        mlir::NVVM::CacheEvictionPriorityAttr{}, descriptor, mlir::Value(),
        /*tensormap=*/true);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMAPrefetchDescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1 || !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_prefetch_descriptor requires exactly 1 argument: desc";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_prefetch_descriptor desc operand must be a TMA descriptor";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMALoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMALoadFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value mbarId =
        positionalCount > 4
            ? resolved_args[4]
            : mlir::arith::ConstantIndexOp::create(builder, location, 0)
                  .getResult();
    mlir::Value predicate =
        positionalCount > 5
            ? resolved_args[5]
            : mlir::arith::ConstantIntOp::create(builder, location, 1, 1)
                  .getResult();
    mlir::Value multicastMask =
        positionalCount > 6
            ? resolved_args[6]
            : mlir::arith::ConstantIntOp::create(builder, location, -1, 32)
                  .getResult();
    mlir::Value expectTx =
        positionalCount > 7
            ? resolved_args[7]
            : mlir::arith::ConstantIntOp::create(builder, location, 1, 1)
                  .getResult();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "mbar_id") {
            mbarId = value;
        } else if (name == "predicate") {
            predicate = value;
        } else if (name == "multicast_mask") {
            if (value) {
                multicastMask = value;
            }
        } else if (name == "expect_tx") {
            expectTx = value;
        }
    }

    cf::NVVMTMALoadOp::create(
        builder, location,
        mlir::ValueRange{resolved_args[0], resolved_args[1], resolved_args[2],
                         resolved_args[3], mbarId, predicate, multicastMask,
                         expectTx},
        mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMALoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 4 || positionalCount > 8) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load requires dst, desc, coords, barrier and optional "
               "mbar_id, predicate, multicast_mask, expect_tx";
        return false;
    }

    for (auto name : keywords) {
        if (name != "mbar_id" && name != "predicate" &&
            name != "multicast_mask" && name != "expect_tx") {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "tma_load got unsupported keyword argument '" << name << "'";
            return false;
        }
    }

    auto requireArg = [&](size_t index, llvm::StringRef name) -> bool {
        if (index >= resolved_args.size() || !resolved_args[index]) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name << " operand for tma_load";
            return false;
        }
        return true;
    };

    if (!requireArg(0, "dst") || !requireArg(1, "desc") ||
        !requireArg(2, "coords") || !requireArg(3, "barrier")) {
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load dst operand must be a memref";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load desc operand must be a TMA descriptor";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[3].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load barrier operand must be of type mbarrier_group_t";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "tma_load " << name << " operand must be an integer type";
            return false;
        }
        return true;
    };

    for (size_t i = 4; i < positionalCount; ++i) {
        if (!checkIntArg(resolved_args[i], i == 4   ? "mbar_id"
                                           : i == 5 ? "predicate"
                                           : i == 6 ? "multicast_mask"
                                                    : "expect_tx")) {
            return false;
        }
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        auto name = llvm::StringRef(keywords[i]);
        auto value = resolved_args[positionalCount + i];
        if (name == "mbar_id" || name == "predicate" || name == "expect_tx" ||
            (name == "multicast_mask" && value)) {
            if (!checkIntArg(value, name)) {
                return false;
            }
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMAStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMAStoreFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value predicate =
        positionalCount > 3
            ? resolved_args[3]
            : mlir::arith::ConstantIntOp::create(builder, location, 1, 1)
                  .getResult();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "predicate") {
            predicate = value;
        }
    }

    cf::NVVMTMAStoreOp::create(builder, location,
                               mlir::ValueRange{resolved_args[0],
                                                resolved_args[1],
                                                resolved_args[2], predicate},
                               mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMAStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 3 || positionalCount > 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store requires src, desc, coords and optional predicate";
        return false;
    }

    for (auto name : keywords) {
        if (name != "predicate") {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "tma_store got unsupported keyword argument '" << name
                << "'";
            return false;
        }
    }

    auto requireArg = [&](size_t index, llvm::StringRef name) -> bool {
        if (index >= resolved_args.size() || !resolved_args[index]) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name << " operand for tma_store";
            return false;
        }
        return true;
    };

    if (!requireArg(0, "src") || !requireArg(1, "desc") ||
        !requireArg(2, "coords")) {
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store src operand must be a memref";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store desc operand must be a TMA descriptor";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "tma_store " << name << " operand must be an integer type";
            return false;
        }
        return true;
    };

    if (positionalCount > 3 && !checkIntArg(resolved_args[3], "predicate")) {
        return false;
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        if (!checkIntArg(resolved_args[positionalCount + i], keywords[i])) {
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateGlobalAtomicAddFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckGlobalAtomicAddFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto pointer = createPointerFromMemRef(
        builder, location, resolved_args[2], resolved_args[0],
        mlir::NVVM::NVVMMemorySpace::Global);
    if (!pointer) {
        return nullptr;
    }
    auto atomic = mlir::LLVM::AtomicRMWOp::create(
        builder, location, mlir::LLVM::AtomicBinOp::add, pointer,
        resolved_args[1], mlir::LLVM::AtomicOrdering::monotonic, "device");
    return atomic.getResult();
}

mlir::Value NVVMIntrinsic::CreateStoreGlobalV4U32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckStoreGlobalV4U32Function(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto pointer = createPointerFromMemRef(
        builder, location, resolved_args[0], resolved_args[1],
        mlir::NVVM::NVVMMemorySpace::Global);
    llvm::SmallVector<mlir::Value> operands{pointer};
    for (int64_t index = 0; index < 4; ++index) {
        operands.push_back(mlir::vector::ExtractOp::create(
            builder, location, resolved_args[2], index));
    }
    mlir::LLVM::InlineAsmOp::create(
        builder, location, mlir::TypeRange{}, operands,
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}", /*hasSideEffects=*/true,
        /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value NVVMIntrinsic::CreateLoadSharedV4U32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckLoadSharedV4U32Function(call_expr, ctx, resolved_args)) {
        return nullptr;
    }
    auto pointer = createPointerFromMemRef(
        builder, location, resolved_args[0], resolved_args[1],
        mlir::NVVM::NVVMMemorySpace::Shared);
    llvm::SmallVector<mlir::Type> elementTypes(4, builder.getI32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, mlir::ValueRange{pointer},
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}", /*hasSideEffects=*/true,
        /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    auto vectorType = mlir::VectorType::get({4}, builder.getI32Type());
    auto zero = mlir::arith::ConstantIntOp::create(builder, location, 0, 32);
    mlir::Value result = mlir::vector::BroadcastOp::create(
        builder, location, vectorType, zero);
    for (int64_t index = 0; index < 4; ++index) {
        auto value = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getI32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{index});
        result = mlir::vector::InsertOp::create(builder, location, value,
                                                result, index);
    }
    return result;
}

mlir::Value NVVMIntrinsic::CreateLoadSharedV2F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckLoadSharedV2F32Function(call_expr, ctx, resolved_args)) {
        return nullptr;
    }
    auto pointer = createPointerFromMemRef(
        builder, location, resolved_args[0], resolved_args[1],
        mlir::NVVM::NVVMMemorySpace::Shared);
    llvm::SmallVector<mlir::Type> elementTypes(2, builder.getF32Type());
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);
    auto inlineAsm = mlir::LLVM::InlineAsmOp::create(
        builder, location, structType, mlir::ValueRange{pointer},
        "ld.shared.v2.f32 {$0, $1}, [$2];", "=f,=f,r,~{memory}",
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    auto vectorType = mlir::VectorType::get({2}, builder.getF32Type());
    auto zero = mlir::arith::ConstantFloatOp::create(
        builder, location, builder.getF32Type(), llvm::APFloat(0.0f));
    mlir::Value result = mlir::vector::BroadcastOp::create(
        builder, location, vectorType, zero);
    for (int64_t index = 0; index < 2; ++index) {
        auto value = mlir::LLVM::ExtractValueOp::create(
            builder, location, builder.getF32Type(), inlineAsm.getRes(),
            llvm::ArrayRef<int64_t>{index});
        result = mlir::vector::InsertOp::create(builder, location, value,
                                                result, index);
    }
    return result;
}

mlir::Value NVVMIntrinsic::CreateCopySharedToGlobalV4U32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCopySharedToGlobalV4U32Function(call_expr, ctx, resolved_args)) {
        return nullptr;
    }
    auto globalPointer = createPointerFromMemRef(
        builder, location, resolved_args[0], resolved_args[1],
        mlir::NVVM::NVVMMemorySpace::Global);
    auto sharedPointer = createPointerFromMemRef(
        builder, location, resolved_args[2], resolved_args[3],
        mlir::NVVM::NVVMMemorySpace::Shared);
    mlir::LLVM::InlineAsmOp::create(
        builder, location, mlir::TypeRange{},
        mlir::ValueRange{globalPointer, sharedPointer},
        "{ .reg .b32 data<4>;\n"
        "ld.shared.v4.u32 {data0, data1, data2, data3}, [$1];\n"
        "st.global.v4.u32 [$0], {data0, data1, data2, data3};\n}",
        "l,r,~{memory}", /*hasSideEffects=*/true,
        /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value NVVMIntrinsic::CreateDrainKDAOutputFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckDrainKDAOutputFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto zero = createDefaultIndex(builder, location);
    auto globalPointer = createPointerFromMemRef(
        builder, location, resolved_args[0], zero,
        mlir::NVVM::NVVMMemorySpace::Global);
    auto sharedPointer = createPointerFromMemRef(
        builder, location, resolved_args[1], zero,
        mlir::NVVM::NVVMMemorySpace::Shared);
    llvm::SmallVector<mlir::Value> operands{globalPointer, sharedPointer};
    for (size_t index = 2; index < 5; ++index) {
        operands.push_back(castIntegerTo(builder, location,
                                         resolved_args[index],
                                         builder.getI32Type()));
    }
    const int64_t heads = *getConstantIntValue(resolved_args[5]);
    const int64_t secondHalfBytes = heads * 8 * 128 * 2;
    std::string asmString =
        "{ .reg .b32 token, channel, offset, saddr, data<4>; "
        ".reg .b64 goffset, gaddr;\n"
        "shr.u32 token, $3, 4;\n"
        "and.b32 channel, $3, 15;\n"
        "shl.b32 channel, channel, 4;\n"
        "mad.lo.u32 offset, $2, 16, token;\n"
        "mad.lo.u32 offset, offset, " +
        std::to_string(heads) +
        ", $4;\n"
        "shl.b32 offset, offset, 8;\n"
        "add.u32 offset, offset, channel;\n"
        "cvt.u64.u32 goffset, offset;\n"
        "add.u64 gaddr, $0, goffset;\n"
        "mad.lo.u32 saddr, token, 272, channel;\n"
        "add.u32 saddr, $1, saddr;\n"
        "ld.shared.v4.u32 {data0, data1, data2, data3}, [saddr];\n"
        "st.global.v4.u32 [gaddr], {data0, data1, data2, data3};\n"
        "ld.shared.v4.u32 {data0, data1, data2, data3}, [saddr+2176];\n"
        "st.global.v4.u32 [gaddr+" +
        std::to_string(secondHalfBytes) +
        "], {data0, data1, data2, data3};\n}";
    mlir::LLVM::InlineAsmOp::create(
        builder, location, mlir::TypeRange{}, operands, asmString,
        "l,r,r,r,r,~{memory}", /*hasSideEffects=*/true,
        /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value NVVMIntrinsic::CreateStoreKDAFactorBlockFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!CheckStoreKDAFactorBlockFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto materializeVector = [&](mlir::Value value) -> mlir::Value {
        if (mlir::isa<mlir::VectorType>(value.getType())) {
            return value;
        }
        auto vectorType = mlir::VectorType::get({4}, builder.getF32Type());
        auto zero = mlir::arith::ConstantIndexOp::create(builder, location, 0);
        return cf::AveLangMemRefLoadVecOp::create(
                   builder, location, vectorType, value,
                   mlir::ValueRange{zero})
            .getResult();
    };
    auto result = materializeVector(resolved_args[6]);
    llvm::SmallVector<mlir::Value> operands;
    for (size_t index = 0; index < 6; ++index) {
        operands.push_back(castIntegerTo(builder, location,
                                         resolved_args[index],
                                         builder.getI32Type()));
    }
    for (int64_t element = 0; element < 4; ++element) {
        operands.push_back(mlir::vector::ExtractOp::create(
            builder, location, result, element));
    }

    const int64_t side = *getConstantIntValue(resolved_args[7]);
    const int64_t kind = *getConstantIntValue(resolved_args[8]);
    std::string asmString =
        "{ .reg .b32 channel, tmp, addr, packed; .reg .f32 lo, hi;\n"
        "shl.b32 channel, $3, 5;\n"
        "shl.b32 tmp, $2, 4;\n"
        "add.u32 channel, channel, tmp;\n"
        "shl.b32 tmp, $5, 1;\n"
        "add.u32 channel, channel, tmp;\n";
    if (side != 0) {
        asmString += "add.u32 channel, channel, 8;\n";
    }
    llvm::SmallVector<int64_t> offsets;
    if (kind == 0) {
        asmString +=
            "shr.u32 addr, channel, 3;\n"
            "shl.b32 addr, addr, 8;\n"
            "shl.b32 tmp, $4, 4;\n"
            "add.u32 addr, addr, tmp;\n"
            "and.b32 tmp, channel, 7;\n"
            "shl.b32 tmp, tmp, 1;\n"
            "add.u32 addr, addr, tmp;\n"
            "mad.lo.u32 tmp, $1, 4096, $0;\n"
            "add.u32 addr, addr, tmp;\n";
        offsets = {0, 128};
    } else {
        asmString +=
            "mad.lo.u32 addr, $4, 272, channel;\n"
            "mad.lo.u32 tmp, $1, 4352, $0;\n"
            "add.u32 addr, addr, tmp;\n"
            "add.u32 addr, addr, 26112;\n";
        offsets = {0, 2176};
    }
    for (int64_t pair = 0; pair < 2; ++pair) {
        const int64_t lo = 6 + 2 * pair;
        const int64_t hi = lo + 1;
        if (kind == 0) {
            asmString += "neg.f32 lo, $" + std::to_string(lo) +
                         ";\nneg.f32 hi, $" + std::to_string(hi) +
                         ";\ncvt.rn.bf16x2.f32 packed, hi, lo;\n";
        } else {
            asmString += "cvt.rn.bf16x2.f32 packed, $" +
                         std::to_string(hi) + ", $" +
                         std::to_string(lo) + ";\n";
        }
        asmString += "st.shared.u32 [addr+" +
                     std::to_string(offsets[pair]) + "], packed;\n";
    }
    asmString += "}";
    mlir::LLVM::InlineAsmOp::create(
        builder, location, mlir::TypeRange{}, operands, asmString,
        "r,r,r,r,r,r,f,f,f,f,~{memory}",
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckStoreGlobalV4U32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    bool valid = resolved_args.size() == 3 && resolved_args[0] &&
                 isMemRefLike(resolved_args[0].getType()) &&
                 resolved_args[1] &&
                 resolved_args[1].getType().isIntOrIndex();
    auto vectorType =
        valid ? mlir::dyn_cast<mlir::VectorType>(resolved_args[2].getType())
              : mlir::VectorType{};
    valid = valid && vectorType && vectorType.getRank() == 1 &&
            vectorType.getDimSize(0) == 4 &&
            vectorType.getElementType().isInteger(32);
    if (!valid) {
        ctx->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            call_expr->GetSourceRange().getBegin())
            << "store_global_v4_u32 requires a destination tensor, byte "
               "offset, and vector<4xi32> value";
    }
    return valid;
}

bool NVVMIntrinsic::CheckLoadSharedV4U32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    bool valid = resolved_args.size() == 2 && resolved_args[0] &&
                 isMemRefLike(resolved_args[0].getType()) &&
                 resolved_args[1] &&
                 resolved_args[1].getType().isIntOrIndex();
    if (valid) {
        auto memrefType =
            mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
        auto gpuSpace = mlir::gpu::AddressSpaceAttr::get(
            ctx->GetCurrentFunctionGenerator()->GetBuilder().getContext(),
            mlir::gpu::AddressSpace::Workgroup);
        valid = memrefType && memrefType.getMemorySpace() == gpuSpace;
    }
    if (!valid) {
        ctx->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            call_expr->GetSourceRange().getBegin())
            << "load_shared_v4_u32 requires a shared tensor and byte offset";
    }
    return valid;
}

bool NVVMIntrinsic::CheckLoadSharedV2F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    bool valid = resolved_args.size() == 2 && resolved_args[0] &&
                 isMemRefLike(resolved_args[0].getType()) &&
                 resolved_args[1] &&
                 resolved_args[1].getType().isIntOrIndex();
    if (valid) {
        auto memrefType =
            mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
        auto gpuSpace = mlir::gpu::AddressSpaceAttr::get(
            ctx->GetCurrentFunctionGenerator()->GetBuilder().getContext(),
            mlir::gpu::AddressSpace::Workgroup);
        valid = memrefType && memrefType.getMemorySpace() == gpuSpace &&
                memrefType.getElementType().isF32();
    }
    if (!valid) {
        ctx->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            call_expr->GetSourceRange().getBegin())
            << "load_shared_v2_f32 requires a shared f32 tensor and byte "
               "offset";
    }
    return valid;
}

bool NVVMIntrinsic::CheckCopySharedToGlobalV4U32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    bool valid = resolved_args.size() == 4 && resolved_args[0] &&
                 isMemRefLike(resolved_args[0].getType()) &&
                 resolved_args[1] &&
                 resolved_args[1].getType().isIntOrIndex() &&
                 resolved_args[2] &&
                 isMemRefLike(resolved_args[2].getType()) &&
                 resolved_args[3] &&
                 resolved_args[3].getType().isIntOrIndex();
    if (valid) {
        auto memrefType =
            mlir::dyn_cast<cf::MemRefType>(resolved_args[2].getType());
        auto gpuSpace = mlir::gpu::AddressSpaceAttr::get(
            ctx->GetCurrentFunctionGenerator()->GetBuilder().getContext(),
            mlir::gpu::AddressSpace::Workgroup);
        valid = memrefType && memrefType.getMemorySpace() == gpuSpace;
    }
    if (!valid) {
        ctx->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            call_expr->GetSourceRange().getBegin())
            << "copy_shared_to_global_v4_u32 requires global/shared tensors "
               "and byte offsets";
    }
    return valid;
}

bool NVVMIntrinsic::CheckDrainKDAOutputFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    bool valid = resolved_args.size() == 6 && resolved_args[0] &&
                 isMemRefLike(resolved_args[0].getType()) &&
                 resolved_args[1] && isMemRefLike(resolved_args[1].getType());
    if (valid) {
        auto memrefType =
            mlir::dyn_cast<cf::MemRefType>(resolved_args[1].getType());
        auto gpuSpace = mlir::gpu::AddressSpaceAttr::get(
            ctx->GetCurrentFunctionGenerator()->GetBuilder().getContext(),
            mlir::gpu::AddressSpace::Workgroup);
        valid = memrefType && memrefType.getMemorySpace() == gpuSpace;
        for (size_t index = 2; index < 5; ++index) {
            valid = valid && resolved_args[index] &&
                    resolved_args[index].getType().isIntOrIndex();
        }
        auto heads = getConstantIntValue(resolved_args[5]);
        valid = valid && heads && *heads > 0;
    }
    if (!valid) {
        ctx->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            call_expr->GetSourceRange().getBegin())
            << "drain_kda_output requires global/shared tensors, pending, "
               "thread, and head integer indices, and a positive constant "
               "head count";
    }
    return valid;
}

bool NVVMIntrinsic::CheckStoreKDAFactorBlockFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto isF32x4 = [&](mlir::Value value) {
        if (!value)
            return false;
        if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(value.getType())) {
            return vectorType.getRank() == 1 &&
                   vectorType.getNumElements() == 4 &&
                   vectorType.getElementType().isF32();
        }
        if (auto memrefType = mlir::dyn_cast<cf::MemRefType>(value.getType())) {
            return memrefType.getRank() == 1 &&
                   memrefType.getShape()[0] == 4 &&
                   memrefType.getElementType() == builder.getF32Type();
        }
        return false;
    };
    bool valid = resolved_args.size() == 9;
    if (valid) {
        for (size_t index = 0; index < 6; ++index) {
            valid = valid && resolved_args[index] &&
                    resolved_args[index].getType().isIntOrIndex();
        }
        valid = valid && isF32x4(resolved_args[6]);
        auto side = getConstantIntValue(resolved_args[7]);
        auto kind = getConstantIntValue(resolved_args[8]);
        valid = valid && side && (*side == 0 || *side == 1) && kind &&
                (*kind == 0 || *kind == 1);
    }
    if (!valid) {
        ctx->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            call_expr->GetSourceRange().getBegin())
            << "store_kda_factor_block requires six integer indices, a 4xf32 "
               "value, and constant side/kind values (0 or 1)";
    }
    return valid;
}

bool NVVMIntrinsic::CheckGlobalAtomicAddFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "atomic_add requires exactly 3 arguments: byte_offset, value, "
               "tensor";
        return false;
    }
    for (auto value : resolved_args) {
        if (!value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate operands for atomic_add";
            return false;
        }
    }
    if (!resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "atomic_add byte_offset must be an integer";
        return false;
    }
    if (!resolved_args[1].getType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "atomic_add currently supports only i32 values";
        return false;
    }
    auto tensorType = mlir::dyn_cast<cf::MemRefType>(resolved_args[2].getType());
    if (!tensorType || !tensorType.getElementType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "atomic_add requires a global i32 tensor";
        return false;
    }
    if (auto addressSpace =
            mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
                tensorType.getMemorySpace());
        addressSpace &&
        addressSpace.getValue() != mlir::gpu::AddressSpace::Global) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "atomic_add requires global memory";
        return false;
    }
    return true;
}

// Factory function to create NVVM intrinsic module
std::unique_ptr<NamedModule> CreateNVVMIntrinsicModule() {
    return std::make_unique<NVVMIntrinsic>();
}

} // namespace causalflow::avelang::ir
