import os
import re
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from typing import Callable
from typing import Optional
from threading import Lock
from ..i18n import translate


__all__ = [
    "CoLeanRechecker",
]


class CoLeanRechecker:
    
    # ----------------------------- CoLeanRechecker 初始化 ----------------------------- 
    
    def __init__(
        self,
        keyword: str = "Claim",
    )-> None:
        
        self._lock: Lock = Lock()
        
        self._keyword: str = keyword
        
        self._revalidator_name_to_func: \
            Dict[str, Callable[[str, List[str]], bool]] = {}
            
        self._last_invalid_cause: str = ""
        
    # ----------------------------- 外部动作 ----------------------------- 
    
    def add_revalidators(
        self,
        revalidators: List[Tuple[str, Callable[[str, List[str]], bool]]],
    )-> None:
        
        with self._lock:
            
            for revalidator_name, revalidator_func in revalidators:
                
                self._revalidator_name_to_func[revalidator_name] = \
                    revalidator_func
                    
                    
    def remove_revalidator(
        self,
        revalidator_name: str,
    )-> None:
        
        with self._lock:
            
            if revalidator_name not in self._revalidator_name_to_func:
                
                raise KeyError(
                    translate("CoLeanRechecker 未储存 %s ，删除出错！") % (revalidator_name)
                )
                
            else:
                del self._revalidator_name_to_func[revalidator_name]
                
                
    def revalidate(
        self,
        lean_code: str,
        mode: Literal["file", "string"] = "file",
        encoding: str = "UTF-8",
    )-> bool:
        
        with self._lock:
        
            if mode == "file":
                
                lean_code = self._revalidate_get_file_content(
                    file_path = lean_code,
                    encoding = encoding,
                )
                    
            lean_code = re.sub(r'--.*?$', '', lean_code, flags=re.MULTILINE)
            lean_code = re.sub(r'/-(.|\n)*?-/','', lean_code, flags=re.DOTALL)
                    
            axiom_pattern = re.compile(r'axiom\s+(\w+)')
            
            for m in axiom_pattern.finditer(lean_code):
                
                ident = m.group(1)
                
                if ident != self._keyword:
                    
                    self._last_invalid_cause = translate(
                        "在允许引入的公理 %s 以外，lean code 中出现了公理 %s ，CoLean 系统无法保证其正确性！"
                    ) % (self._keyword, ident)
                    
                    return False

            axiom_spans = [m.span() for m in axiom_pattern.finditer(lean_code)]
            keyword_pattern = re.compile(rf'\b{self._keyword}\b')

            for m in keyword_pattern.finditer(lean_code):
                
                pos = m.start()
                
                if any(start <= pos < end for start, end in axiom_spans):
                    continue

                tail_text = lean_code[pos:]
                result = self._revalidate_extract_claim_parts(
                    text = tail_text, 
                    start_pos = len(self._keyword),
                )

                if result is None:
                    self._last_invalid_cause = translate(
                        "在位置 %d 发现关键字 %s 后，未能找到符合格式的推理外包逻辑！"
                    ) % (pos, self._keyword)
                    return False

                prop, verified_facts_raw, revalidator_name = result

                prop_pattern = re.compile(r'\{prop\s*:=\s*([^,}]+)')
                verified_props = prop_pattern.findall(verified_facts_raw)

                if revalidator_name not in self._revalidator_name_to_func:
                    
                    self._last_invalid_cause = translate(
                        "验证器 %s 未知！"
                    ) % revalidator_name
                    
                    return False

                func = self._revalidator_name_to_func[revalidator_name]

                if not func(prop, verified_props):
                    
                    self._last_invalid_cause = translate(
                        "验证器 %s 复核命题 %s 失败！"
                    ) % (revalidator_name, prop)
                    
                    return False

            return True
    
    
    def get_invalid_cause(
        self
    )-> str:
        
        with self._lock:
            return self._last_invalid_cause
        
    # ----------------------------- 内部动作 ----------------------------- 
    
    def _revalidate_get_file_content(
        self,
        file_path: str,
        encoding: str,
    )-> str:
        
        if not file_path.strip():
                    
            raise ValueError(
                translate("CoLeanRechecker revalidate 时出错：文件路径为空！")
            )

        abs_path = os.path.abspath(file_path)
        
        if not os.path.exists(abs_path):
            
            raise FileNotFoundError(
                translate("CoLeanRechecker revalidate 时出错：文件 %s 不存在！") 
                % abs_path
            )
        
        if not os.path.isfile(abs_path):
            
            raise IsADirectoryError(
                translate("CoLeanRechecker revalidate 时出错：路径 %s 不是文件！") 
                % abs_path
            )
        
        if not os.access(abs_path, os.R_OK):
            
            raise PermissionError(
                translate("CoLeanRechecker revalidate 时出错：无权限读取文件 %s ！") 
                % abs_path
            )

        try:
            
            with open(
                file = abs_path, 
                mode = "r", 
                encoding = encoding,
            ) as file_pointer:
                
                return file_pointer.read()
                
        except Exception as error:
            
            raise IOError(
                translate("CoLeanRechecker revalidate 时出错：读取文件 %s 时出错 %s") 
                % (abs_path, str(error))
            )
            
            
    def _revalidate_extract_claim_parts(
        self, 
        text: str, 
        start_pos: int
    )-> Optional[Tuple[str, str, str]]:
        
        def skip_whitespace(i):
            
            while i < len(text) and text[i].isspace():
                i += 1
                
            return i

        def parse_balanced(i, open_char, close_char):
            
            assert text[i] == open_char
            
            depth = 1
            i += 1
            start = i
            
            while i < len(text) and depth > 0:
                
                if text[i] == open_char:
                    depth += 1
                elif text[i] == close_char:
                    depth -= 1
                    
                i += 1
                
            if depth != 0:
                return None, i
            
            return text[start:i - 1], i

        i = start_pos
        i = skip_whitespace(i)

        if i >= len(text):
            return None
        
        if text[i] == "(":
            
            prop, i = parse_balanced(i, "(", ")")
            if prop is None: return None
            
        else:
            
            start = i
            
            while i < len(text) and not text[i].isspace() and text[i] not in ['[', '"']:
                i += 1
                
            prop = text[start:i]

        i = skip_whitespace(i)

        if i >= len(text) or text[i] != "[":
            return None
        
        verified, i = parse_balanced(i, "[", "]")
        if verified is None: return None

        i = skip_whitespace(i)

        if i >= len(text) or text[i] != '"':
            return None
        
        i += 1
        start = i
        
        while i < len(text) and text[i] != '"':
            i += 1
        if i >= len(text):
            return None
        
        revalidator = text[start:i]
        
        return prop.strip(), verified.strip(), revalidator.strip()
        
