#pragma once

#include <functional>
#include <string>
#include <string_view>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir {
class ModuleOp;
class OpBuilder;
class MLIRContext;
} // namespace mlir

namespace causalflow::avelang::ir::intrinsics {

inline std::string MakeIntrinsicFuncName(std::string_view modulePrefix,
                                         std::string_view baseName) {
    llvm::SmallString<64> name("_avelang_");
    name.append(modulePrefix.begin(), modulePrefix.end());
    name.push_back('_');
    name.append(baseName.begin(), baseName.end());
    return std::string(name);
}

mlir::ModuleOp GetOwningModule(mlir::OpBuilder &builder);
mlir::ModuleOp GetOrCreateImplementationContainer(mlir::ModuleOp parent,
                                                  std::string_view backend,
                                                  std::string_view libraryPath);

mlir::LogicalResult EnsureIntrinsicDeclarations(
    mlir::ModuleOp module, llvm::StringRef libraryName,
    llvm::StringRef libraryContents,
    const std::function<void(mlir::MLIRContext *)> &dialectLoader);

} // namespace causalflow::avelang::ir::intrinsics
