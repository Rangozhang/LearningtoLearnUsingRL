       �K"	   �Q*�Abrain.Event:2h�[��3      V�T	��:�Q*�A"�g
N
PlaceholderPlaceholder*
dtype0*
shape: *
_output_shapes
:
N
	cost/tagsConst*
dtype0*
valueB
 Bcost*
_output_shapes
: 
N
costScalarSummary	cost/tagsPlaceholder*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummarycost*
_output_shapes
: *
N
N
Placeholder_1Placeholder*
dtype0*
shape: *
_output_shapes
: 
`
Placeholder_2Placeholder*
dtype0*
shape: *(
_output_shapes
:����������
_
Placeholder_3Placeholder*
dtype0*
shape: *'
_output_shapes
:���������

f
model_var/zerosConst*
dtype0*
valueB	�
*    *
_output_shapes
:	�

�
model_var/VariableVariable*
dtype0*
shape:	�
*
	container *
shared_name *
_output_shapes
:	�

�
model_var/Variable/AssignAssignmodel_var/Variablemodel_var/zeros*
validate_shape(*%
_class
loc:@model_var/Variable*
use_locking(*
T0*
_output_shapes
:	�

�
model_var/Variable/readIdentitymodel_var/Variable*%
_class
loc:@model_var/Variable*
T0*
_output_shapes
:	�

^
model_var/zeros_1Const*
dtype0*
valueB
*    *
_output_shapes
:

~
model_var/Variable_1Variable*
dtype0*
shape:
*
	container *
shared_name *
_output_shapes
:

�
model_var/Variable_1/AssignAssignmodel_var/Variable_1model_var/zeros_1*
validate_shape(*'
_class
loc:@model_var/Variable_1*
use_locking(*
T0*
_output_shapes
:

�
model_var/Variable_1/readIdentitymodel_var/Variable_1*'
_class
loc:@model_var/Variable_1*
T0*
_output_shapes
:

�
MatMulMatMulPlaceholder_2model_var/Variable/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������

_
addAddMatMulmodel_var/Variable_1/read*
T0*'
_output_shapes
:���������

I
SoftmaxSoftmaxadd*
T0*'
_output_shapes
:���������

E
LogLogSoftmax*
T0*'
_output_shapes
:���������

P
mulMulPlaceholder_3Log*
T0*'
_output_shapes
:���������

W
Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
q
SumSummulSum/reduction_indices*#
_output_shapes
:���������*
T0*
	keep_dims( *

Tidx0
=
NegNegSum*
T0*#
_output_shapes
:���������
O
ConstConst*
dtype0*
valueB: *
_output_shapes
:
V
MeanMeanNegConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
\
gradients/Mean_grad/ShapeShapeNeg*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
^
gradients/Mean_grad/Shape_1ShapeNeg*
out_type0*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
{
gradients/Mean_grad/floordivDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
h
gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
T0*#
_output_shapes
:���������
[
gradients/Sum_grad/ShapeShapemul*
out_type0*
T0*
_output_shapes
:
Y
gradients/Sum_grad/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
n
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*
_output_shapes
: 
o
gradients/Sum_grad/modModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*
_output_shapes
: 
]
gradients/Sum_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
`
gradients/Sum_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
`
gradients/Sum_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*
_output_shapes
:
_
gradients/Sum_grad/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
{
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*
_output_shapes
: 
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*#
_output_shapes
:���������*
T0*
N
^
gradients/Sum_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*#
_output_shapes
:���������
}
gradients/Sum_grad/floordivDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*
_output_shapes
:
�
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������

e
gradients/mul_grad/ShapeShapePlaceholder_3*
out_type0*
T0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
out_type0*
T0*
_output_shapes
:
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
m
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:���������

�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
y
gradients/mul_grad/mul_1MulPlaceholder_3gradients/Sum_grad/Tile*
T0*'
_output_shapes
:���������

�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*'
_output_shapes
:���������

�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*'
_output_shapes
:���������

�
gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������

�
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������

t
gradients/Softmax_grad/mulMulgradients/Log_grad/mulSoftmax*
T0*'
_output_shapes
:���������

v
,gradients/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*#
_output_shapes
:���������*
T0*
	keep_dims( *

Tidx0
u
$gradients/Softmax_grad/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
gradients/Softmax_grad/ReshapeReshapegradients/Softmax_grad/Sum$gradients/Softmax_grad/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/Softmax_grad/subSubgradients/Log_grad/mulgradients/Softmax_grad/Reshape*
T0*'
_output_shapes
:���������

z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:���������

^
gradients/add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:
*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Softmax_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
gradients/add_grad/Sum_1Sumgradients/Softmax_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������

�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
:

�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencymodel_var/Variable/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder_2+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�

n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*(
_output_shapes
:����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�

�
>GradientDescent/update_model_var/Variable/ApplyGradientDescentApplyGradientDescentmodel_var/VariablePlaceholder_10gradients/MatMul_grad/tuple/control_dependency_1*%
_class
loc:@model_var/Variable*
use_locking( *
T0*
_output_shapes
:	�

�
@GradientDescent/update_model_var/Variable_1/ApplyGradientDescentApplyGradientDescentmodel_var/Variable_1Placeholder_1-gradients/add_grad/tuple/control_dependency_1*'
_class
loc:@model_var/Variable_1*
use_locking( *
T0*
_output_shapes
:

�
GradientDescentNoOp?^GradientDescent/update_model_var/Variable/ApplyGradientDescentA^GradientDescent/update_model_var/Variable_1/ApplyGradientDescent
F
initNoOp^model_var/Variable/Assign^model_var/Variable_1/Assign"쏵��>      �*;v	�.?�Q*�AJ�}
��
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyGradientDescent
var"T�

alpha"T

delta"T
out"T�"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
9
Div
x"T
y"T
z"T"
Ttype:
2	
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype
.
Identity

input"T
output"T"	
Ttype
+
Log
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
2
Mod
x"T
y"T
z"T"
Ttype:
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
Variable
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*0.12.12v0.12.0-10-g4d924e7-dirty�g
N
PlaceholderPlaceholder*
dtype0*
shape: *
_output_shapes
:
N
	cost/tagsConst*
dtype0*
valueB
 Bcost*
_output_shapes
: 
N
costScalarSummary	cost/tagsPlaceholder*
T0*
_output_shapes
: 
I
Merge/MergeSummaryMergeSummarycost*
N*
_output_shapes
: 
N
Placeholder_1Placeholder*
dtype0*
shape: *
_output_shapes
: 
`
Placeholder_2Placeholder*
dtype0*
shape: *(
_output_shapes
:����������
_
Placeholder_3Placeholder*
dtype0*
shape: *'
_output_shapes
:���������

f
model_var/zerosConst*
dtype0*
valueB	�
*    *
_output_shapes
:	�

�
model_var/VariableVariable*
dtype0*
shape:	�
*
shared_name *
	container *
_output_shapes
:	�

�
model_var/Variable/AssignAssignmodel_var/Variablemodel_var/zeros*
validate_shape(*%
_class
loc:@model_var/Variable*
use_locking(*
T0*
_output_shapes
:	�

�
model_var/Variable/readIdentitymodel_var/Variable*%
_class
loc:@model_var/Variable*
T0*
_output_shapes
:	�

^
model_var/zeros_1Const*
dtype0*
valueB
*    *
_output_shapes
:

~
model_var/Variable_1Variable*
dtype0*
shape:
*
shared_name *
	container *
_output_shapes
:

�
model_var/Variable_1/AssignAssignmodel_var/Variable_1model_var/zeros_1*
validate_shape(*'
_class
loc:@model_var/Variable_1*
use_locking(*
T0*
_output_shapes
:

�
model_var/Variable_1/readIdentitymodel_var/Variable_1*'
_class
loc:@model_var/Variable_1*
T0*
_output_shapes
:

�
MatMulMatMulPlaceholder_2model_var/Variable/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������

_
addAddMatMulmodel_var/Variable_1/read*
T0*'
_output_shapes
:���������

I
SoftmaxSoftmaxadd*
T0*'
_output_shapes
:���������

E
LogLogSoftmax*
T0*'
_output_shapes
:���������

P
mulMulPlaceholder_3Log*
T0*'
_output_shapes
:���������

W
Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
q
SumSummulSum/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:���������
=
NegNegSum*
T0*#
_output_shapes
:���������
O
ConstConst*
dtype0*
valueB: *
_output_shapes
:
V
MeanMeanNegConst*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
\
gradients/Mean_grad/ShapeShapeNeg*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
^
gradients/Mean_grad/Shape_1ShapeNeg*
out_type0*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
{
gradients/Mean_grad/floordivDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
h
gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
T0*#
_output_shapes
:���������
[
gradients/Sum_grad/ShapeShapemul*
out_type0*
T0*
_output_shapes
:
Y
gradients/Sum_grad/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
n
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*
_output_shapes
: 
o
gradients/Sum_grad/modModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*
_output_shapes
: 
]
gradients/Sum_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
`
gradients/Sum_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
`
gradients/Sum_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*
_output_shapes
:
_
gradients/Sum_grad/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
{
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*
_output_shapes
: 
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
N*
T0*#
_output_shapes
:���������
^
gradients/Sum_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*#
_output_shapes
:���������
}
gradients/Sum_grad/floordivDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*
_output_shapes
:
�
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/DynamicStitch*
Tshape0*
T0*
_output_shapes
:
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������

e
gradients/mul_grad/ShapeShapePlaceholder_3*
out_type0*
T0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
out_type0*
T0*
_output_shapes
:
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
m
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:���������

�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������

y
gradients/mul_grad/mul_1MulPlaceholder_3gradients/Sum_grad/Tile*
T0*'
_output_shapes
:���������

�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������

g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*'
_output_shapes
:���������

�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*'
_output_shapes
:���������

�
gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������

�
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������

t
gradients/Softmax_grad/mulMulgradients/Log_grad/mulSoftmax*
T0*'
_output_shapes
:���������

v
,gradients/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:���������
u
$gradients/Softmax_grad/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
gradients/Softmax_grad/ReshapeReshapegradients/Softmax_grad/Sum$gradients/Softmax_grad/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:���������
�
gradients/Softmax_grad/subSubgradients/Log_grad/mulgradients/Softmax_grad/Reshape*
T0*'
_output_shapes
:���������

z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:���������

^
gradients/add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB:
*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Softmax_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������

�
gradients/add_grad/Sum_1Sumgradients/Softmax_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:

g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*'
_output_shapes
:���������

�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
:

�
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencymodel_var/Variable/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder_2+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�

n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*(
_output_shapes
:����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�

�
>GradientDescent/update_model_var/Variable/ApplyGradientDescentApplyGradientDescentmodel_var/VariablePlaceholder_10gradients/MatMul_grad/tuple/control_dependency_1*%
_class
loc:@model_var/Variable*
use_locking( *
T0*
_output_shapes
:	�

�
@GradientDescent/update_model_var/Variable_1/ApplyGradientDescentApplyGradientDescentmodel_var/Variable_1Placeholder_1-gradients/add_grad/tuple/control_dependency_1*'
_class
loc:@model_var/Variable_1*
use_locking( *
T0*
_output_shapes
:

�
GradientDescentNoOp?^GradientDescent/update_model_var/Variable/ApplyGradientDescentA^GradientDescent/update_model_var/Variable_1/ApplyGradientDescent
F
initNoOp^model_var/Variable/Assign^model_var/Variable_1/Assign""
train_op

GradientDescent"�
	variables��
L
model_var/Variable:0model_var/Variable/Assignmodel_var/Variable/read:0
R
model_var/Variable_1:0model_var/Variable_1/Assignmodel_var/Variable_1/read:0"
	summaries


cost:0"�
trainable_variables��
L
model_var/Variable:0model_var/Variable/Assignmodel_var/Variable/read:0
R
model_var/Variable_1:0model_var/Variable_1/Assignmodel_var/Variable_1/read:0�Je       �K"	���Q*�A*

costTP�?<�yw       ��-	ܻ2�Q*�A*

costt6*?����       ��-	����Q*�A*

cost�?\��       ��-	�?��Q*�A*

cost�@�>���       ��-	�Z�Q*�A*

cost�Q�>���	       ��-	f�\�Q*�A*

costɗ�>ե�       ��-	���Q*�A*

cost[��>����       ��-	.��Q*�A*

cost� �>*/��       ��-	l�
�Q*�A*

costۄ�>��,�       ��-	g�L�Q*�A	*

cost���>�d_�       ��-	w�R*�A
*

cost���>0&J_       ��-	��R*�A*

cost
��>�_��       ��-	��R*�A*

cost���>�Η�       ��-	.X�R*�A*

cost#�>�7       ��-	��R*�A*

cost��>��݂       ��-	�~dR*�A*

cost���>�{�       ��-	Fy�"R*�A*

cost(��>����       ��-	�V�'R*�A*

cost��>ৌt       ��-	���,R*�A*

cost�Y�>��{�       ��-	�}	2R*�A*

cost�װ>���       ��-	��97R*�A*

cost�v�>Kx14       ��-	��S<R*�A*

cost�1�>�g�       ��-	�]AR*�A*

cost"�>6/
^       ��-	�ƤFR*�A*

cost4�>O�G       ��-	lPLR*�A*

costMت>���       ��-	�YlQR*�A*

cost4ө>1��       ��-	t�>VR*�A*

costh�>�?s~       ��-	)�,[R*�A*

cost��>R�9)       ��-	j�'`R*�A*

cost�7�>.ہP       ��-	*�eR*�A*

costpf�>���       ��-	��kR*�A*

cost[��>�S�        ��-	)��qR*�A*

cost���>���       ��-	�'owR*�A *

costD�>�W��       ��-	�$}R*�A!*

cost㚣>4�6�       ��-	!s��R*�A"*

cost	��>��)�       ��-	D�]�R*�A#*

cost�Y�>����       ��-	g&C�R*�A$*

cost�ˡ>�4��       ��-	��ĔR*�A%*

costZ=�>Ʉ�O       ��-	����R*�A&*

cost.��>ӟʡ       ��-	�{�R*�A'*

cost�,�>NRy]       ��-	��p�R*�A(*

cost���>����       ��-	�`=�R*�A)*

cost<5�>Kͩ�       ��-	c�.�R*�A**

cost���>��_       ��-	���R*�A+*

cost<L�>�͵�       ��-	��R*�A,*

cost�ޝ>��Zz       ��-	����R*�A-*

costns�>�줫       ��-	��f�R*�A.*

costG�>Y��l       ��-	"�m�R*�A/*

costH��>:b9;       ��-	�i#�R*�A0*

cost�E�>�5�       ��-	���R*�A1*

cost��>�d/�       ��-	'0��R*�A2*

cost\��>���       ��-	�T��R*�A3*

costi)�>�۹       ��-		��R*�A4*

cost*ښ>��'T       ��-	1D(�R*�A5*

cost%��>����       ��-	�D �R*�A6*

cost�,�>�P8N       ��-	���R*�A7*

cost�ޙ> 6�       ��-	��>S*�A8*

costQ��>��8g       ��-	~
S*�A9*

cost�A�>M�)�       ��-	���S*�A:*

cost���>Z=R       ��-	QڂS*�A;*

cost5��>��$       ��-	���S*�A<*

costJj�>��@       ��-	�s
$S*�A=*

costm.�>j���       ��-	��+*S*�A>*

costX�>���       ��-	cE�0S*�A?*

costz��>� �       ��-	C�M8S*�A@*

costBe�>�Rz`       ��-	jߣ?S*�AA*

cost}%�>afL       ��-	r�FS*�AB*

cost��>�)Z�       ��-	A�%NS*�AC*

cost.��>�z��       ��-	#�bUS*�AD*

cost�m�>۶�N       ��-	f��\S*�AE*

cost�,�>?���       ��-	ծ�cS*�AF*

cost'��>ut�       ��-	�@jS*�AG*

cost�Õ>�=P�       ��-	0��pS*�AH*

cost1��>0�)       ��-	ȕwS*�AI*

cost�Y�>�-�       ��-	��6~S*�AJ*

cost��>0�[�       ��-	y#��S*�AK*

costl�>�ٻm       ��-	̋S*�AL*

cost���>E�)       ��-	� z�S*�AM*

cost���>`��&       ��-	~�*�S*�AN*

costZ�>�ٟ       ��-	���S*�AO*

costL,�>��p�