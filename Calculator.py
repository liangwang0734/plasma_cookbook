"""Generic Evaluator/Calculator"""

__all__ = ["Calculator"]


import re
import numpy as np

try:
    import numexpr as ne

    HAVE_NUMEXRP = True
except ImportError:
    HAVE_NUMEXRP = False

################################
# GENERIC EVALUATOR/CALCULATOR #
################################


def evaluate(expr, get, known_fields, consts=None, **kwargs):
    """Evaluate an expression using a given field retriever.

    - Adapted from: https://viscid-hub.github.io/Viscid-docs/docs/dev/_modules/viscid/calculator/evaluator.html
    - Not salting the symbols, since only local variables and numexpr-supported
        numpy core functions are used.

    Args:
        expr : An expr which consists of known fields and parameters.
        get: A function that retrieve the data for a field in known_fields.
        known_fields: A list of variable names that the get function knows how
            to get.
        consts: A dictionary of named scalars, like 'epsilon0', 'mu0', 'm',
            'gamma', mapped to their values.
        **kwargs : Keyword arguments to be passed to the get function.

    Returns:
        result : Data corresponding to the expression.
    """
    # match legit variable names like a, a3, a3_, a_3, _a3, etc., and skip
    # illegal names like 9a, 9_a
    symbol_pattern = r"\b([_A-Za-z][_a-zA-Z0-9]*)\b"
    # do not match function calls like 'sqrt('
    symbol_pattern += r"(?!\s*\()"

    # dict: symbol -> corresponding data
    local_dict = {}

    if consts is None:
        consts = {}
    else:
        assert isinstance(consts, dict)
        consts = consts.copy()

    def cache_symbol_data(symbols):
        symbol = symbols.groups()[0]

        is_known_field = symbol in known_fields
        is_known_const = symbol in consts

        if not (is_known_field or is_known_const):
            msg = (
                f'"{symbol}" is not a recognized field/constant.'
                + f" Known fields include {known_fields}."
                + f" Known constants include {list(consts.keys())}."
            ).replace("'", "")
            raise KeyError(msg)

        if symbol not in local_dict:
            if symbol in consts:
                symbol_data = consts[symbol]
            else:
                symbol_data = get(symbol, **kwargs)
            local_dict[symbol] = symbol_data

        return symbol

    re.sub(symbol_pattern, cache_symbol_data, expr)

    result = ne.evaluate(
        expr,
        local_dict=local_dict,
        global_dict={},
    )

    return result


class Calculator:
    def __init__(self, getter, native_fields, derived_fields=None, consts=None):
        """Create a calculator.

        Args:
        getter: A function that retrieve the data for a field in known_fields.
        known_fields: A list of field names that the getter knows how to get.
        derived_fields: A dictionary (or mapping) of derived fields defined as
            expressions containing known fields.
        consts: A dictionary of named scalars, like 'epsilon0', 'mu0', 'm',
            'gamma'.
        """
        self.getter = getter
        self.native_fields = native_fields
        if derived_fields is None:
            self.derived_fields = {}
        else:
            assert isinstance(derived_fields, dict)
            self.derived_fields = derived_fields.copy()
        if consts is None:
            self.consts = {}
        else:
            assert isinstance(consts, dict)
            self.consts = consts.copy()

    @property
    def known_fields(self):
        """Always retrieving the updated known_fields."""
        return self.native_fields + list(self.derived_fields)

    def get(self, expr, **kwargs):
        if expr in self.native_fields:
            return self.getter(expr, **kwargs)
        elif not HAVE_NUMEXRP:
            msg = f"{expr} is not a native field"
            msg += "; numexpr is needed to calculate it."
            raise ValueError()
        elif expr in self.derived_fields:
            return evaluate(
                self.derived_fields[expr],
                self.get,
                self.known_fields,
                self.consts,
                **kwargs,
            )
        else:
            return evaluate(
                expr, self.get, self.known_fields, self.consts, **kwargs
            )

    def __getitem__(self, expr):
        return self.get(expr)
