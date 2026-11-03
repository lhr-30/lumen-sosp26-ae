#include "intrinsic_support.h"

#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/Support/MemoryBuffer.h>

namespace causalflow::avelang::ir::intrinsics {

mlir::ModuleOp GetOwningModule(mlir::OpBuilder &builder) {
    auto *block = builder.getInsertionBlock();
    if (!block)
        return nullptr;
    auto *parent_op = block->getParentOp();
    if (!parent_op)
        return nullptr;
    return parent_op->getParentOfType<mlir::ModuleOp>();
}

mlir::ModuleOp
GetOrCreateImplementationContainer(mlir::ModuleOp parent,
                                   std::string_view backend,
                                   std::string_view libraryPath) {
    if (!parent)
        return nullptr;

    auto context = parent.getContext();
    std::string moduleName = "_avelang_intrinsics_" + std::string(backend);
    if (auto existing = parent.lookupSymbol<mlir::ModuleOp>(moduleName)) {
        if (!libraryPath.empty() &&
            !existing->hasAttr("ave.intrinsic_library")) {
            mlir::OpBuilder builder(context);
            existing->setAttr("ave.intrinsic_library",
                              builder.getStringAttr(libraryPath));
        }
        return existing;
    }

    mlir::OpBuilder builder(context);
    builder.setInsertionPointToEnd(parent.getBody());
    auto implModule = mlir::ModuleOp::create(builder, builder.getUnknownLoc(),
                                             builder.getStringAttr(moduleName));
    implModule->setAttr("ave.intrinsic_impl", builder.getStringAttr(backend));
    if (!libraryPath.empty()) {
        implModule->setAttr("ave.intrinsic_library",
                            builder.getStringAttr(libraryPath));
    }
    return implModule;
}

mlir::LogicalResult EnsureIntrinsicDeclarations(
    mlir::ModuleOp module, llvm::StringRef libraryName,
    llvm::StringRef libraryContents,
    const std::function<void(mlir::MLIRContext *)> &dialectLoader) {
    if (!module)
        return mlir::failure();

    auto *context = module.getContext();
    if (dialectLoader)
        dialectLoader(context);

    auto processLibrary = [&](mlir::ModuleOp library) -> mlir::LogicalResult {
        mlir::OpBuilder builder(context);
        builder.setInsertionPointToStart(module.getBody());
        auto loc = builder.getUnknownLoc();

        for (auto func : library.getOps<mlir::func::FuncOp>()) {
            if (module.lookupSymbol<mlir::func::FuncOp>(func.getName()))
                continue;

            auto declaration = mlir::func::FuncOp::create(
                builder, loc, func.getName(), func.getFunctionType());
            declaration.setPrivate();
        }

        return mlir::success();
    };

    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        libraryContents, libraryName, /*RequiresNullTerminator=*/false);
    auto bufferRef = buffer->getMemBufferRef();

    if (!mlir::isBytecode(bufferRef)) {
        module.emitError() << "intrinsic library '" << libraryName
                           << "' is not MLIR bytecode";
        return mlir::failure();
    }

    mlir::ParserConfig parserConfig(context);
    mlir::Block block;
    if (mlir::failed(mlir::readBytecodeFile(bufferRef, &block, parserConfig))) {
        module.emitError() << "failed to parse intrinsic bytecode '"
                           << libraryName << "'";
        return mlir::failure();
    }

    bool sawModule = false;
    for (mlir::Operation &op : block) {
        auto libraryModule = llvm::dyn_cast<mlir::ModuleOp>(&op);
        if (!libraryModule)
            continue;
        sawModule = true;
        if (mlir::failed(processLibrary(libraryModule)))
            return mlir::failure();
    }

    if (!sawModule) {
        module.emitError() << "intrinsic bytecode '" << libraryName
                           << "' did not contain a module";
        return mlir::failure();
    }

    return mlir::success();
}

} // namespace causalflow::avelang::ir::intrinsics
