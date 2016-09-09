from sympy.core import Symbol, S, Expr, Tuple, Equality, Function
from sympy.core.compatibility import StringIO, is_sequence
from sympy.abc import x, y, z
from sympy.printing.codeprinter import AssignmentError
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)
from dendrite.codegen.glsl_printer import glslcode, GLSLCodePrinter

# The following classes were specialized from Sympy's CodeGen module, as the design of that module is
# ah how we say not so great you see yes?

class Routine(object):
  def __init__(self, name, arguments, result, local_vars, global_vars):
    """Initialize a Routine instance.

    Parameters
    ==========

    name : string
        Name of the routine.

    arguments : list of Arguments
        These are things that appear in arguments of a routine, often
        appearing on the right-hand side of a function call.

    result : Result
        This is the return value of the routine, often appearing on
        the left-hand side of a function call.

    local_vars : list of Symbols
        These are used internally by the routine.

    global_vars : list of Symbols
        Variables which will not be passed into the function.

    """
    self.name = name
    self.arguments = arguments
    self.result = result
    self.local_vars = local_vars
    self.global_vars = global_vars

  def __str__(self):
    return self.__class__.__name__ + "({name!r}, {arguments}, {result}, {local_vars}, {global_vars})".format(**self.__dict__)

  __repr__ = __str__

  @property
  def variables(self):
    """Returns a set of all variables possibly used in the routine.

    For routines with unnamed return values, the dummies that may or
    may not be used will be included in the set.

    """
    v = set(self.local_vars)
    for arg in self.arguments:
      v.add(arg.name)
    v.add(self.result.result_var)
    return v

def get_default_datatype(expr):
  """Derives an appropriate datatype based on the expression."""
  if expr.is_integer:
    return "int"
  elif isinstance(expr, MatrixSymbol):
    if expr.shape[1] == 1:
      return "vec"+str(expr.shape[0])
    elif expr.shape[1] == expr.shape[0]:
      return "mat"+str(expr.shape[0])
    else:
      raise CodeGenError("Data shape %s not supported" % expr.shape)
  else:
    return "float"

class Variable(object):
  """Represents a typed variable."""

  def __init__(self, name, datatype=None):
    """Return a new variable.

    Parameters
    ==========

    name : Symbol or MatrixSymbol

    datatype : optional
        When not given, the data type will be guessed based on the
        assumptions on the symbol argument.

    """
    if not isinstance(name, (Symbol, MatrixSymbol)):
      raise TypeError("The first argument must be a sympy symbol.")
    if datatype is None:
      datatype = get_default_datatype(name)

    self._name = name
    self.datatype = datatype

  def __str__(self):
    return "%s(%r)" % (self.__class__.__name__, self.name)

  __repr__ = __str__

  @property
  def name(self):
    return self._name

class Argument(Variable):
  """An abstract Argument data structure: a name and a data type.

  This structure is refined in the descendants below.

  """
  pass

class InputArgument(Argument):
  pass

class ResultBase(object):
  """Base class for all "outgoing" information from a routine.

  Objects of this class stores a sympy expression, and a sympy object
  representing a result variable that will be used in the generated code
  only if necessary.

  """
  def __init__(self, expr, result_var):
    self.expr = expr
    self.result_var = result_var

  def __str__(self):
    return "%s(%r, %r)" % (self.__class__.__name__, self.expr,
        self.result_var)

  __repr__ = __str__

class Result(Variable, ResultBase):
  """An expression for a return value.

  The name result is used to avoid conflicts with the reserved word
  "return" in the python language.  It is also shorter than ReturnValue.

  These may or may not need a name in the destination (e.g., "return(x*y)"
  might return a value without ever naming it).

  """

  def __init__(self, expr, name=None, result_var=None, datatype=None, dimensions=None, precision=None):
    """Initialize a return value.

    Parameters
    ==========

    expr : SymPy expression

    name : Symbol, MatrixSymbol, optional
        The name of this return variable.  When used for code generation,
        this might appear, for example, in the prototype of function in a
        list of return values.  A dummy name is generated if omitted.

    result_var : Symbol, Indexed, optional
        Something that can be used to assign a value to this variable.
        Typically the same as `name` but for Indexed this should be e.g.,
        "y[i]" whereas `name` should be the Symbol "y".  Defaults to
        `name` if omitted.

    datatype : optional
        When not given, the data type will be guessed based on the
        assumptions on the symbol argument.

    dimension : sequence containing tupes, optional
        If present, this variable is interpreted as an array,
        where this sequence of tuples specifies (lower, upper)
        bounds for each index of the array.

    """
    if not isinstance(expr, (Expr, Tuple)):
      raise TypeError("The first argument must be a sympy expression or tuple.")

    if name is None:
      name = 'result_%d' % abs(hash(expr))

    if isinstance(expr, Tuple):
      # This specifies the output is of vector type
      # TODO: Add matrix type
      name = MatrixSymbol(name, len(expr), 1)
    else:
      name = Symbol(name)

    if result_var is None:
      result_var = name

    Variable.__init__(self, name, datatype=datatype)
    ResultBase.__init__(self, expr, result_var)

class CodeGen(object):
  """Abstract class for the code generators."""

  def __init__(self):
    pass

  def routine(self, name, expr, argument_sequence, global_vars):
    """Creates an Routine object that is appropriate for this language.

    Here, we assume at most one return value (the l-value) which must be
    scalar. If ``argument_sequence`` is None, arguments will
    be ordered alphabetically.
    """

    if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
      if not expr:
        raise ValueError("No expression given")
      expression = Tuple(*expr)
    else:
      expression = expr

    # local variables
    local_vars = {i.label for i in expression.atoms(Idx)}

    # global variables
    global_vars = set() if global_vars is None else set(global_vars)

    # symbols that should be arguments
    symbols = expression.free_symbols - local_vars - global_vars
    spatial_symbols = set([x,y,z])
    if len(spatial_symbols & symbols) > 0:
      # Use vec3 variable p instead of x,y,z
      symbols |= set([MatrixSymbol("p", 3, 1)])
      symbols -= spatial_symbols
    new_symbols = set([])
    new_symbols.update(symbols)

    for symbol in symbols:
      if isinstance(symbol, Idx):
        new_symbols.remove(symbol)
        new_symbols.update(symbol.args[1].free_symbols)
    symbols = new_symbols

    arg_list = []

    # setup input argument list
    array_symbols = {}
    for array in expression.atoms(Indexed):
      array_symbols[array.base.label] = array
    for array in expression.atoms(MatrixSymbol):
      array_symbols[array] = array

    for symbol in sorted(symbols, key=str):
      if symbol in array_symbols:
        dims = []
        array = array_symbols[symbol]
        for dim in array.shape:
          dims.append((S.Zero, dim - 1))
        metadata = {'dimensions': dims}
      else:
        metadata = {}

      arg_list.append(InputArgument(symbol, **metadata))

    if argument_sequence is not None:
      # if the user has supplied IndexedBase instances, we'll accept that
      new_sequence = []
      for arg in argument_sequence:
        if isinstance(arg, IndexedBase):
          new_sequence.append(arg.label)
        else:
          new_sequence.append(arg)
      argument_sequence = new_sequence

      missing = [x for x in arg_list if x.name not in argument_sequence]
      if missing:
        msg = "Argument list didn't specify: {0} "
        msg = msg.format(", ".join([str(m.name) for m in missing]))
        raise CodeGenArgumentListError(msg, missing)

      # create redundant arguments to produce the requested sequence
      name_arg_dict = {x.name: x for x in arg_list}
      new_args = []
      for symbol in argument_sequence:
        try:
          new_args.append(name_arg_dict[symbol])
        except KeyError:
          new_args.append(InputArgument(symbol))
      arg_list = new_args

    return_val = Result(expression)
    return Routine(name, arg_list, return_val, local_vars, global_vars)

  def write(self, routines, empty=True):
    """Writes all the source code files for the given routines.

    The generated source is returned as a list of (filename, contents)
    tuples, or is written to files (see below).

    Parameters
    ==========

    routines : list
        A list of Routine instances to be written

    empty : bool, optional
        When True, empty lines are included to structure the source
        files. [default: True]

    """
    contents = StringIO()
    self.dump_fn(routines, contents, empty)
    return contents.getvalue()

  def dump_code(self, routines, f, empty=True):
    """Write the code by calling language specific methods.

    Parameters
    ==========

    routines : list
        A list of Routine instances.

    f : file-like
        Where to write the file.

    empty : bool, optional
        When True, empty lines are included to structure the source
        files.  [default : True]
    """

    code_lines = []

    for routine in routines:
      if empty:
        code_lines.append("\n")
      code_lines.extend(self._get_routine_opening(routine))
      if empty:
        code_lines.append("\n")
      code_lines.extend(self._call_printer(routine))
      if empty:
        code_lines.append("\n")
      code_lines.extend(self._get_routine_ending(routine))

    code_lines = self._indent_code(''.join(code_lines))

    if code_lines:
      f.write(code_lines)

class CodeGenError(Exception):
  pass

class CodeGenArgumentListError(Exception):
  @property
  def missing_args(self):
    return self.args[1]

header_comment = """Code generated with sympy %(version)s"""

def glslcodegen(name_expr, empty=True, argument_sequence=None, global_vars=None):
  # Initialize the code generator.
  code_gen = GLSLCodeGen()

  if isinstance(name_expr[0], str):
    # single tuple is given, turn it into a singleton list with a tuple.
    name_expr = [name_expr]

  routines = []
  for name, expr in name_expr:
    routines.append(code_gen.routine(name, expr, argument_sequence, global_vars))

  return code_gen.write(routines, empty)

class GLSLCodeGen(CodeGen):
  def get_prototype(self, routine):
    if isinstance(routine.result, Tuple):
      glsltype = "vec"+str(len(routine.result))
    else:
      glsltype = routine.result.datatype

    type_args = []
    for arg in routine.arguments:
      name = glslcode(arg.name)
      type_args.append((arg.datatype, name))
    arguments = ", ".join(["%s %s" % t for t in type_args])
    return "%s %s(%s)" % (glsltype, routine.name, arguments)

  def _get_routine_opening(self, routine):
    prototype = self.get_prototype(routine)
    return ["%s {\n" % prototype]

  def _call_printer(self, routine):
    code_lines = []

    result = routine.result
    assign_to = routine.name + "_result"
    code_lines.append("{0} {1};\n".format(result.datatype, str(assign_to)))

    try:
      constants, not_supported, glsl_expr = glslcode(result.expr, human=False, assign_to=assign_to)
    except AssignmentError:
      assign_to = result.result_var
      code_lines.append("%s %s;\n" % (result, str(assign_to)))
      constants, not_supported, glsl_expr = glslcode(result.expr, human=False, assign_to=assign_to)

    for name, value in sorted(constants, key=str):
      code_lines.append("#define %s = %s;\n" % (name, value))

    code_lines.append("%s\n" % glsl_expr)
    code_lines.append("   return %s;\n" % assign_to)
    return code_lines

  def _indent_code(self, codelines):
    p = GLSLCodePrinter()
    return p.indent_code(codelines)

  def _get_routine_ending(self, routine):
    return ["}\n"]

  def dump_glsl(self, routines, f, empty=True):
    self.dump_code(routines, f, empty)

  dump_glsl.__doc__ = CodeGen.dump_code.__doc__

  dump_fn = dump_glsl
