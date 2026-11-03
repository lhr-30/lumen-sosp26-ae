module {

  func.func private @_avelang_nvvm_mma_16x8x16_f16_f16(%arg0: vector<8xf16>, %arg1: vector<4xf16>, %arg2: vector<4xf16>) -> vector<4xf16> attributes {func.inline = "always"} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    
    %3 = vector.extract_strided_slice %arg0 {offsets = [0], sizes = [2], strides = [1]} : vector<8xf16> to vector<2xf16>
    %4 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<8xf16> to vector<2xf16>
    %5 = vector.extract_strided_slice %arg0 {offsets = [4], sizes = [2], strides = [1]} : vector<8xf16> to vector<2xf16>
    %6 = vector.extract_strided_slice %arg0 {offsets = [6], sizes = [2], strides = [1]} : vector<8xf16> to vector<2xf16>
    
    %7 = vector.extract_strided_slice %arg1 {offsets = [0], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
    %8 = vector.extract_strided_slice %arg1 {offsets = [2], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
    
    // Extract 2xf16 slices from arg2 (4xf16)
    %9 = vector.extract_strided_slice %arg2 {offsets = [0], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
    %10 = vector.extract_strided_slice %arg2 {offsets = [2], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
    
    %11 = nvvm.mma.sync A[%3, %4, %5, %6]  B[%7, %8]  C[%9, %10]  {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    
    %12 = llvm.extractvalue %11[0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    %13 = llvm.extractvalue %11[1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    
    %cst = arith.constant 0.000000e+00 : f16
    %14 = vector.broadcast %cst : f16 to vector<4xf16>
    %15 = vector.insert_strided_slice %12, %14 {offsets = [0], sizes = [2], strides = [1]} : vector<2xf16> into vector<4xf16>
    %16 = vector.insert_strided_slice %13, %15 {offsets = [2], sizes = [2], strides = [1]} : vector<2xf16> into vector<4xf16>
    
    return %16 : vector<4xf16>
  }

  func.func private @_avelang_nvvm_mma_16x8x8_f16_f32(%arg0: vector<4xf16>, %arg1: vector<2xf16>, %arg2: vector<4xf32>) -> vector<4xf32> attributes {func.inline = "always"} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index

    %3 = vector.extract_strided_slice %arg0 {offsets = [0], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
    %4 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
    %5 = vector.extract_strided_slice %arg1 {offsets = [0], sizes = [2], strides = [1]} : vector<2xf16> to vector<2xf16>
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %idx2 = arith.constant 2 : index
    %idx3 = arith.constant 3 : index
    %cval0 = vector.extract %arg2[%idx0] : f32 from vector<4xf32>
    %cval1 = vector.extract %arg2[%idx1] : f32 from vector<4xf32>
    %cval2 = vector.extract %arg2[%idx2] : f32 from vector<4xf32>
    %cval3 = vector.extract %arg2[%idx3] : f32 from vector<4xf32>
    %6 = nvvm.mma.sync A[%3, %4]  B[%5]  C[%cval0, %cval1, %cval2, %cval3]  {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 8>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
    %res0 = llvm.extractvalue %6[0] : !llvm.struct<(f32, f32, f32, f32)>
    %res1 = llvm.extractvalue %6[1] : !llvm.struct<(f32, f32, f32, f32)>
    %res2 = llvm.extractvalue %6[2] : !llvm.struct<(f32, f32, f32, f32)>
    %res3 = llvm.extractvalue %6[3] : !llvm.struct<(f32, f32, f32, f32)>
    %zero = arith.constant 0.0 : f32
    %vec0 = vector.broadcast %zero : f32 to vector<4xf32>
    %vec1 = vector.insert %res0, %vec0[%idx0] : f32 into vector<4xf32>
    %vec2 = vector.insert %res1, %vec1[%idx1] : f32 into vector<4xf32>
    %vec3 = vector.insert %res2, %vec2[%idx2] : f32 into vector<4xf32>
    %vec4 = vector.insert %res3, %vec3[%idx3] : f32 into vector<4xf32>
    return %vec4 : vector<4xf32>
  }

  func.func private @_avelang_nvvm_ldmatrix_m8n8_x1_b16(%arg0: index) -> i32 attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %result = nvvm.ldmatrix %llvm_ptr {layout = #nvvm.mma_layout<row>, num = 1 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : (!llvm.ptr<3>) -> i32
    return %result : i32
  }

  func.func private @_avelang_nvvm_ldmatrix_m8n8_x1_b16_trans(%arg0: index) -> i32 attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %result = nvvm.ldmatrix %llvm_ptr {layout = #nvvm.mma_layout<col>, num = 1 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : (!llvm.ptr<3>) -> i32
    return %result : i32
  }

  func.func private @_avelang_nvvm_ldmatrix_m8n8_x2_b16(%arg0: index) -> !llvm.struct<(i32, i32)> attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %result = nvvm.ldmatrix %llvm_ptr {layout = #nvvm.mma_layout<row>, num = 2 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
    return %result : !llvm.struct<(i32, i32)>
  }

  func.func private @_avelang_nvvm_ldmatrix_m8n8_x2_b16_trans(%arg0: index) -> !llvm.struct<(i32, i32)> attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %result = nvvm.ldmatrix %llvm_ptr {layout = #nvvm.mma_layout<col>, num = 2 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>
    return %result : !llvm.struct<(i32, i32)>
  }

  func.func private @_avelang_nvvm_ldmatrix_m8n8_x4_b16(%arg0: index) -> !llvm.struct<(i32, i32, i32, i32)> attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %result = nvvm.ldmatrix %llvm_ptr {layout = #nvvm.mma_layout<row>, num = 4 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    return %result : !llvm.struct<(i32, i32, i32, i32)>
  }

  func.func private @_avelang_nvvm_ldmatrix_m8n8_x4_b16_trans(%arg0: index) -> !llvm.struct<(i32, i32, i32, i32)> attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %result = nvvm.ldmatrix %llvm_ptr {layout = #nvvm.mma_layout<col>, num = 4 : i32, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
    return %result : !llvm.struct<(i32, i32, i32, i32)>
  }

  func.func private @_avelang_nvvm_stmatrix_m8n8_x1_b16(%arg0: index, %arg1: i32) -> () attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    nvvm.stmatrix %llvm_ptr, %arg1 {layout = #nvvm.mma_layout<row>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : !llvm.ptr<3>, i32
    return
  }

  func.func private @_avelang_nvvm_stmatrix_m8n8_x1_b16_trans(%arg0: index, %arg1: i32) -> () attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    nvvm.stmatrix %llvm_ptr, %arg1 {layout = #nvvm.mma_layout<col>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : !llvm.ptr<3>, i32
    return
  }

  func.func private @_avelang_nvvm_stmatrix_m8n8_x2_b16(%arg0: index, %arg1: vector<2xi32>) -> () attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %idx0 = arith.constant 0 : index
    %val0 = vector.extract %arg1[%idx0] : i32 from vector<2xi32>
    %idx1 = arith.constant 1 : index
    %val1 = vector.extract %arg1[%idx1] : i32 from vector<2xi32>
    nvvm.stmatrix %llvm_ptr, %val0, %val1 {layout = #nvvm.mma_layout<row>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : !llvm.ptr<3>, i32, i32
    return
  }

  func.func private @_avelang_nvvm_stmatrix_m8n8_x2_b16_trans(%arg0: index, %arg1: vector<2xi32>) -> () attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %idx0 = arith.constant 0 : index
    %val0 = vector.extract %arg1[%idx0] : i32 from vector<2xi32>
    %idx1 = arith.constant 1 : index
    %val1 = vector.extract %arg1[%idx1] : i32 from vector<2xi32>
    nvvm.stmatrix %llvm_ptr, %val0, %val1 {layout = #nvvm.mma_layout<col>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : !llvm.ptr<3>, i32, i32
    return
  }

  func.func private @_avelang_nvvm_stmatrix_m8n8_x4_b16(%arg0: index, %arg1: vector<4xi32>) -> () attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %idx0 = arith.constant 0 : index
    %val0 = vector.extract %arg1[%idx0] : i32 from vector<4xi32>
    %idx1 = arith.constant 1 : index
    %val1 = vector.extract %arg1[%idx1] : i32 from vector<4xi32>
    %idx2 = arith.constant 2 : index
    %val2 = vector.extract %arg1[%idx2] : i32 from vector<4xi32>
    %idx3 = arith.constant 3 : index
    %val3 = vector.extract %arg1[%idx3] : i32 from vector<4xi32>
    nvvm.stmatrix %llvm_ptr, %val0, %val1, %val2, %val3 {layout = #nvvm.mma_layout<row>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : !llvm.ptr<3>, i32, i32, i32, i32
    return
  }

  func.func private @_avelang_nvvm_stmatrix_m8n8_x4_b16_trans(%arg0: index, %arg1: vector<4xi32>) -> () attributes {func.inline = "always"} {
     
    %ptr_i64 = arith.index_cast %arg0 : index to i64
    %llvm_ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr<3>
    %idx0 = arith.constant 0 : index
    %val0 = vector.extract %arg1[%idx0] : i32 from vector<4xi32>
    %idx1 = arith.constant 1 : index
    %val1 = vector.extract %arg1[%idx1] : i32 from vector<4xi32>
    %idx2 = arith.constant 2 : index
    %val2 = vector.extract %arg1[%idx2] : i32 from vector<4xi32>
    %idx3 = arith.constant 3 : index
    %val3 = vector.extract %arg1[%idx3] : i32 from vector<4xi32>
    nvvm.stmatrix %llvm_ptr, %val0, %val1, %val2, %val3 {layout = #nvvm.mma_layout<col>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>, eltType = #nvvm.ld_st_matrix_elt_type<b16>} : !llvm.ptr<3>, i32, i32, i32, i32
    return
  }

}
