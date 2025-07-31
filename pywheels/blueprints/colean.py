import json
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from typing import Callable
from threading import Lock
from ..i18n import translate
from ..file_tools.basic import delete_file
from ..file_tools.basic import get_temp_file_path
from ..task_runner.task_runner import execute_command


__all__ = [
    "LeanProxy",
]


class LeanProxy:
    
    # ----------------------------- LeanProxy 初始化 ----------------------------- 
    
    def __init__(
        self,
    )-> None:
        
        self._lock: Lock = Lock()
        
        self._name_to_proxy_func: \
            Dict[str, Callable[[str, List[str]], bool]] = {}
        
    # ----------------------------- 外部动作 ----------------------------- 
    
    def add_proxies(
        self,
        proxies: List[Tuple[str, Callable[[str, List[str]], bool]]],
    )-> None:
        
        with self._lock:
            
            for proxy_name, proxy_func in proxies:
                
                self._name_to_proxy_func[proxy_name] = \
                    proxy_func
                    
                    
    def remove_proxy(
        self,
        proxy_name: str,
    )-> None:
        
        with self._lock:
            
            if proxy_name not in self._name_to_proxy_func:
                
                raise KeyError(
                    translate("LeanProxy 未储存 %s ，删除出错！") % (proxy_name)
                )
                
            else:
                del self._name_to_proxy_func[proxy_name]
                
                
    def revalidate(
        self,
        lean_code: str,
        mode: Literal["file", "string"] = "file",
        lean_bin: str = "lean",
    )-> bool:
        
        lean_path: str

        if mode == "file":
            lean_path = lean_code
            
        elif mode == "string":
            
            lean_path = get_temp_file_path(
                suffix = ".lean",
                prefix = "tmp_LeanProxyRevalidation_DeleteMe_",
                directory = None,
            )
            
            with open(
                file = lean_path,
                mode = "w",
                encoding = "UTF-8",
            ) as file_pointer:
                
                file_pointer.write(lean_code)
        
        else:
            raise ValueError(
                translate("LeanProxy 复核失败：暂不支持模式 %s！") % (mode)
            )
            
        lean_code_parse_result = execute_command(
            command = f"{lean_bin} --ast {lean_path}"
        )
        
        if not lean_code_parse_result["success"]:
            
            raise RuntimeError(
                translate("LeanProxy 复核失败：lean code 解析失败，请检查是否符合 Lean4 语法。")
            )
            
        lean_code_ast = json.loads(lean_code_parse_result["stdout"])
        
        print(lean_code_ast)
        
        
        if mode == "string": delete_file(lean_path)
        
        return False
    
    
    # ----------------------------- 内部动作 ----------------------------- 
    
    