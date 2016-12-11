"""
GLSL shader code printer

The GLSLCodePrinter converts single sympy expressions into single
GLSL expressions.

"""

from sympy.core import S, symbols, Tuple
from sympy.codegen.ast import Assignment
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
from sympy.core.compatibility import string_types, range

# dictionary mapping sympy function to (argument_conditions, glsl_function).
# Used in GLSLCodePrinter._print_Function(self)
known_functions = {
  'Abs': 'abs',
  'ceiling': 'ceil',
  'Mod': 'mod',
  'floor': 'floor',
  'atan': 'atan',
  'cos': 'cos',
  'sin': 'sin',
  'acos': 'acos',
  'asin': 'asin'
}

class GLSLCodePrinter(CodePrinter):
  """"A Printer to convert python expressions to strings of GLSL code
  """
  printmethod = '_glsl'
  language = 'GLSL'

  _default_settings = {
    'order': None,
    'full_prec': 'auto',
    'precision': 15,
    'user_functions': {},
    'human': True,
    'contract': True
  }

  def __init__(self, settings={}):
    CodePrinter.__init__(self, settings)
    self.known_functions = dict(known_functions)
    userfuncs = settings.get('user_functions', {})
    self.known_functions.update(userfuncs)

  def _rate_index_position(self, p):
    return p*5

  def _get_statement(self, codestring):
    return "%s;" % codestring

  def _get_comment(self, text):
    return "// {0}".format(text)

  def _declare_number_const(self, name, value):
    return "#define {0} = {1};".format(name, value)

  def _format_code(self, lines):
    return self.indent_code(lines)

  def _traverse_matrix_indices(self, mat):
    rows, cols = mat.shape
    return ((i, j) for i in range(rows) for j in range(cols))

  def _get_loop_opening_ending(self, indices):
    open_lines = []
    close_lines = []
    loopstart = "for (int %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){"
    for i in indices:
      open_lines.append(loopstart % {
        'varble': self._print(i.label),
        'start': self._print(i.lower),
        'end': self._print(i.upper + 1)})
      close_lines.append("}")
    return open_lines, close_lines

  def _print_Symbol(self, expr):
    name = super(CodePrinter, self)._print_Symbol(expr)
    if name in ["x", "y", "z"]:
      return "p."+name
    else:
      return super()._print_Symbol(expr)

  def _print_Pow(self, expr):
    PREC = precedence(expr)
    if expr.exp == -1:
      return '1/%s' % (self.parenthesize(expr.base, PREC))
    elif expr.exp == 0.5:
      return 'sqrt(%s)' % self._print(expr.base)
    else:
      # Convert to float, otherwise type error
      return 'pow(%s, float(%s))' % (self._print(expr.base),
                                     self._print(expr.exp))

  def _print_Min(self, expr, **kwargs):
    from sympy import Min
    if len(expr.args) == 1:
      return self._print(expr.args[0], **kwargs)

    return 'min({0}, {1})'.format(
      self._print(expr.args[0], **kwargs),
      self._print(Min(*expr.args[1:]), **kwargs))

  def _print_Max(self, expr, **kwargs):
    from sympy import Max
    if len(expr.args) == 1:
      return self._print(expr.args[0], **kwargs)

    return 'max({0}, {1})'.format(
      self._print(expr.args[0], **kwargs),
      self._print(Max(*expr.args[1:]), **kwargs))

  def _print_Rational(self, expr):
    p, q = int(expr.p), int(expr.q)
    return '%d/%d' % (p, q)

  def _print_Tuple(self, expr):
    return 'vec'+str(len(expr)) + "(" + ", ".join([self._print(sub) for sub in expr]) + ")"

  def _print_Indexed(self, expr):
    # calculate index for 1d array
    dims = expr.shape
    elem = S.Zero
    offset = S.One
    for i in reversed(range(expr.rank)):
      elem += expr.indices[i]*offset
      offset *= dims[i]
    return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

  def _print_Idx(self, expr):
    return self._print(expr.label)

  def _print_Pi(self, expr):
    return '3.1415926535897932384626433832795'

  def _print_Infinity(self, expr):
    return '1./0.'

  def _print_NegativeInfinity(self, expr):
    return '-1./0.'

  def _print_Subs(self, expr):
    # HACK: free symbols should be the names of compiled GLSL functions to call
    g, f = list(expr.free_symbols)
    function_call = self._print(g) + "(" + ", ".join([self._print(a) for a in expr.args[1]]) + ");"
    print(function_call)
    temp_symbols = symbols("gx gy gz")
    assign_to = Tuple(*temp_symbols)
    assign = self._print(Assignment(assign_to, function_call))
    print(assign)
    composed_call = self._print(f) + self._print(temp_symbols)
    return assign + "\n" + composed_call

  def _print_Piecewise(self, expr):
    if expr.args[-1].cond != True:
      # We need the last conditional to be a True, otherwise the resulting
      # function may not return a result.
      raise ValueError("All Piecewise expressions must contain an "
                       "(expr, True) statement to be used as a default "
                       "condition. Without one, the generated "
                       "expression may not evaluate to anything under "
                       "some condition.")
    lines = []
    if expr.has(Assignment):
      for i, (e, c) in enumerate(expr.args):
        if i == 0:
          lines.append("if (%s) {" % self._print(c))
        elif i == len(expr.args) - 1 and c == True:
          lines.append("else {")
        else:
          lines.append("else if (%s) {" % self._print(c))
        code0 = self._print(e)
        lines.append(code0)
        lines.append("}")
      return "\n".join(lines)
    else:
      # The piecewise was used in an expression, need to do inline
      # operators. This has the downside that inline operators will
      # not work for statements that span multiple lines (Matrix or
      # Indexed expressions).
      ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
              for e, c in expr.args[:-1]]
      last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
      return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

  def indent_code(self, code):
    """Accepts a string of code or a list of code lines"""

    if isinstance(code, string_types):
      code_lines = self.indent_code(code.splitlines(True))
      return ''.join(code_lines)

    tab = "   "
    inc_token = ('{', '(', '{\n', '(\n')
    dec_token = ('}', ')')

    code = [ line.lstrip(' \t') for line in code ]

    increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
    decrease = [ int(any(map(line.startswith, dec_token)))
                 for line in code ]

    pretty = []
    level = 0
    for n, line in enumerate(code):
      if line == '' or line == '\n':
        pretty.append(line)
        continue
      level -= decrease[n]
      pretty.append("%s%s" % (tab*level, line))
      level += increase[n]
    return pretty


def glslcode(expr, assign_to=None, **settings):
  """Converts an expr to a string of GLSL code

  Parameters
  ==========

  expr : Expr
      A sympy expression to be converted.
  assign_to : optional
      When given, the argument is used as the name of the variable to which
      the expression is assigned. Can be a string, ``Symbol``,
      ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
      line-wrapping, or for expressions that generate multi-line statements.
  user_functions : dict, optional
      A dictionary where keys are ``FunctionClass`` instances and values are
      their string representations. Alternatively, the dictionary value can
      be a list of tuples i.e. [(argument_test, js_function_string)]. See
      below for examples.
  human : bool, optional
      If True, the result is a single string that may contain some constant
      declarations for the number symbols. If False, the same information is
      returned in a tuple of (symbols_to_declare, not_supported_functions,
      code_text). [default=True].
  """

  return GLSLCodePrinter(settings).doprint(expr, assign_to)

def print_glslcode(expr, **settings):
  """Prints the GLSL representation of the given expression.

     See glslcode for the meaning of the optional arguments.
  """
  print(glslcode(expr, **settings))
