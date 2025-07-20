import re
import ast
import astor
import random
from typing import List
from typing import Tuple
from typing import Callable
from scipy.optimize import minimize
from ..i18n import translate


__all__ = [
    "Ansatz",
]


class Ansatz:
    
    # ----------------------------- Ansatz 初始化 ----------------------------- 
    
    def __init__(
        self,
        expression: str,
        variables: List[str],
        functions: List[str],
        seed: int  = 42,
    )-> None:
        
        self._expression = expression
        self._variables = variables
        self._functions = functions

        self._random_generator = random.Random(seed)
        
        self._check_format()
        
    # ----------------------------- 外部动作 -----------------------------   
        
    def get_param_num(
        self,
    )-> int:
        
        return self._param_num
    
    
    def reduce_to_numeric_ansatz(
        self,
        params: List[float],
    )-> str:
    
        return self._reduce_to_numeric_ansatz(params)
    
    
    def apply_to(
        self,
        numeric_ansatz_user: Callable[[str], float],
        param_ranges: List[Tuple[float, float]],
        trial_num: int,
        mode: str = "random",
        do_minimize: bool = True,
    )-> Tuple[List[float], float]:

        if mode == "random":
            
            return self._apply_to_mode_random(
                numeric_ansatz_user,
                param_ranges,
                trial_num,
                do_minimize,
            )

        elif mode == "optimize":
            
            return self._apply_to_mode_optimize(
                numeric_ansatz_user,
                param_ranges,
                trial_num,
                do_minimize,
            )

        else:
            
            raise ValueError(
                translate(
                    "Ansatz 类的 apply to 动作不支持模式 %s，"
                    "请使用 `random` 或 `optimize`！"
                ) % (mode)
            )
    
    # ----------------------------- 内部动作 ----------------------------- 
    
    def _check_format(
        self,
    )-> None:
        
        ansatz_param_num = _check_ansatz_format(
            expression = self._expression,
            variables = self._variables,
            functions = self._functions,
        )
        
        if not ansatz_param_num:
            
            raise RuntimeError(
                translate("拟设格式有误！")
            )
            
        self._param_num = ansatz_param_num
        
        
    def _reduce_to_numeric_ansatz(
        self,
        params: List[float],
    )-> str:
        
        if len(params) != self._param_num:
            
            raise ValueError(
                translate("提供的参数数量与拟设中所需参数数量不符。")
            )

        param_dict = {
            f"param{i + 1}": value for i, value in enumerate(params)
        }

        tree = ast.parse(self._expression, mode='eval')

        class ParamReplacer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in param_dict:
                    return ast.Constant(param_dict[node.id])
                return node

        transformer = ParamReplacer()
        modified_tree = transformer.visit(tree)
        ast.fix_missing_locations(modified_tree)

        return astor.to_source(modified_tree).strip()
    
    
    def _generate_random_params(
        self,
        param_ranges
    )-> List[float]:
        
        return [
            self._random_generator.uniform(param_ranges[i][0], param_ranges[i][1])
            for i in range(self._param_num)
        ]
        
        
    def _apply_to_mode_random(
        self,
        numeric_ansatz_user: Callable[[str], float],
        param_ranges: List[Tuple[float, float]],
        trial_num: int,
        do_minimize: bool,
    )-> Tuple[List[float], float]:
        
        best_params = []
        best_output = float('inf') if do_minimize else float('-inf')
        
        for _ in range(trial_num):
            
            params = self._generate_random_params(param_ranges)
            
            output = numeric_ansatz_user(
                self._reduce_to_numeric_ansatz(params)
            )
            
            if (output < best_output) == do_minimize:
                
                best_output = output
                best_params = params
            
        return best_params, best_output
    
    
    def _apply_to_mode_optimize(
        self,
        numeric_ansatz_user: Callable[[str], float],
        param_ranges: List[Tuple[float, float]],
        trial_num: int,
        do_minimize: bool
    )-> Tuple[List[float], float]:

        def objective(params: List[float]) -> float:
            expr = self._reduce_to_numeric_ansatz(params)
            value = numeric_ansatz_user(expr)
            return value if do_minimize else -value

        best_params = None
        best_output = float('inf') if do_minimize else float('-inf')

        for _ in range(trial_num):
            
            init_params = self._generate_random_params(param_ranges)

            res = minimize(
                fun = objective,
                x0 = init_params,
                bounds = param_ranges,
                method = 'L-BFGS-B'
            )

            if not res.success:
                continue

            score = res.fun if do_minimize else -res.fun

            if ((score < best_output) if do_minimize else (score > best_output)):
                best_output = score
                best_params = res.x.tolist()

        if best_params is None:
            
            raise RuntimeError(
                translate("优化器在所有初始点上均未成功收敛。")
            )

        return best_params, best_output
    
    
def _check_ansatz_format(
    expression: str,
    variables: List[str],
    functions: List[str],
)-> int:
    
    """
    检查输入的表达式是否符合预定义的拟设（ansatz）格式要求。

    本函数会：
    - 校验表达式中使用的运算符、变量、函数是否符合预定义要求；
    - 确保变量名、函数名、参数名称等符号合法，并且符合语法要求；
    - 校验表达式中的参数是否按规定编号且连续，不允许存在常数。

    参数：
        expression (str): 被检查的数学表达式字符串。
        variables (list[str]): 允许使用的变量名列表，表达式中的变量必须严格来自该列表。
        functions (list[str]): 允许使用的函数名列表，函数名必须为裸函数名，不带模块前缀。

    返回值：
        int: 
            - 如果表达式合法，返回最大参数编号（即 'paramN' 的 N 值）。
            - 如果表达式不合法，返回 0。

    注意：
        本函数会首先对 `variables` 和 `functions` 中的内容进行合法性校验，若包含非法名称（如带模块前缀的函数名），将抛出异常。
    """
    
    identifier_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for name in variables + functions:
        
        if not identifier_pattern.fullmatch(name):
            
            raise ValueError(
                translate("非法标识符名：'%s'，应仅由字母、数字、下划线组成，不能包含点号等")
                % (name)
            )

    if re.search(r"[^\w\s+\-*/(),]", expression):
        return 0

    try:
        tree = ast.parse(expression, mode="eval")
        
    except Exception:
        return 0

    used_names = set()
    used_funcs = set()

    param_indices = set()

    def visit(node):
        
        if isinstance(node, ast.BinOp) or isinstance(node, ast.UnaryOp):
            
            allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
            allowed_unops = (ast.UAdd, ast.USub)
            
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, allowed_binops):
                    raise ValueError(translate("不支持的二元运算符"))
                
            if isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, allowed_unops):
                    raise ValueError(translate("不支持的一元运算符"))
                
            visit(node.operand if isinstance(node, ast.UnaryOp) else node.left)
            
            if isinstance(node, ast.BinOp):
                visit(node.right)
                
        elif isinstance(node, ast.Call):
            
            if not isinstance(node.func, ast.Name):
                raise ValueError(translate("函数调用形式非法"))
            
            func_name = node.func.id
            if func_name not in functions:
                raise ValueError(translate("调用了未注册的函数 '%s'")%(func_name))
            
            used_funcs.add(func_name)
            
            for arg in node.args:
                visit(arg)
                
        elif isinstance(node, ast.Name):
            
            name = node.id
            used_names.add(name)
            
            if name.startswith("param"):
                match = re.fullmatch(r"param([1-9][0-9]*)", name)
                
                if not match:
                    raise ValueError(translate("非法参数名称 '%s'")%(name))
                param_indices.add(int(match.group(1)))
                
            elif name not in variables and name not in functions:
                raise ValueError(translate("使用了非法变量或未注册函数 '%s'")%(name))
            
        elif isinstance(node, ast.Constant):
            raise ValueError(translate("表达式中不允许使用任何常数"))
        
        elif isinstance(node, ast.Expr):
            visit(node.value)
            
        else:
            
            raise ValueError(
                translate("表达式中包含不支持的语法节点类型：%s")
                %(type(node).__name__)
            )

    try:
        visit(tree.body)
        
    except Exception:
        return 0

    if param_indices:
        
        max_index = max(param_indices)
        
        if sorted(param_indices) != list(range(1, max_index + 1)):
            return 0
        
        return max_index
    
    else:
        return 0