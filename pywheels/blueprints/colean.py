import os
import re
from tqdm import tqdm
from typing import Set
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
        claim_keyword: str = "Claim",
        prop_field: str = "prop",
    )-> None:
        
        self._lock: Lock = Lock()
        
        self._claim_keyword = claim_keyword
        self._prop_field = prop_field
        
        self._revalidator_name_to_func: \
            Dict[str, Callable[[str, List[str]], bool]] = {}
        self._whitelist_axioms: Set[str] = set()
            
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
                
    
    def add_whitelist_axioms(
        self,
        whitelist_axioms: List[str],
    )-> None:
        
        with self._lock:
            
            for axiom in whitelist_axioms:
                self._whitelist_axioms.add(axiom)
                
                
    def remove_whitelist_axiom(
        self,
        whitelist_axiom: str,
    )-> None:
        
        with self._lock:
            self._whitelist_axioms.remove(whitelist_axiom)
                
                
    def revalidate(
        self,
        lean_code: str,
        mode: Literal["file", "string"] = "file",
        encoding: str = "UTF-8",
        show_progress: bool = False,
    )-> bool:
        
        with self._lock:
            
            if show_progress: print(translate("[CoLean] 复核开始"))
        
            if mode == "file":
                
                lean_code = self._revalidate_get_file_content(
                    file_path = lean_code,
                    encoding = encoding,
                )
                
            if show_progress: print(translate("[CoLean] 正在删除注释"))
                    
            lean_code = re.sub(r'--.*?$', '', lean_code, flags=re.MULTILINE)
            lean_code = re.sub(r'/-(.|\n)*?-/','', lean_code, flags=re.DOTALL)
            
            if show_progress: print(translate("[CoLean] 正在检查自定义公理"))
                    
            axiom_pattern = re.compile(r'axiom\s+(\w+)')
            axiom_match = axiom_pattern.finditer(lean_code)
            
            for current_match in axiom_match:
                
                ident = current_match.group(1)
                
                if ident != self._claim_keyword and ident not in self._whitelist_axioms:
                    
                    self._last_invalid_cause = translate(
                        "在关键字 %s 和白名单公理之外，lean code 中出现了公理 %s ，CoLean 系统无法保证其正确性！"
                    ) % (self._claim_keyword, ident)
                    
                    if show_progress: print(
                        translate("[CoLean] 复核结束；结果：失败，%s")
                        % (self._last_invalid_cause)
                    )
                    
                    return False
                
            if show_progress: print(translate("[CoLean] 正在检查 Claim Structure"))
                
            claim_structure_pattern = re.compile(
                r"structure\s+(\w+)\s+where\s+"
                r"(\w+)\s*\:\s*Prop\s+(\w+)\s*\:\s*(\w+)\s+axiom\s+"
                f"{self._claim_keyword}"
                r"\s*\(\s*(\w+)\s*\:\s*Prop\s*\)"
                r"\s*\(\s*\w+\s*\:\s*List\s+(\w+)\s*\)"
                r"\s*\(\s*\w+\s*\:\s*String\s*\)"
                r"\s*\:\s*(\w+)"
            )
            matchs = claim_structure_pattern.findall(lean_code)
            
            if len(matchs) != 1:
                
                self._last_invalid_cause = translate(
                    "在 lean code 中匹配到了 %d 个 claim structure（定义关键字 %s 的结构），但应有且仅有一个！"
                ) % (len(matchs), self._claim_keyword)
                
                if show_progress: print(
                    translate("[CoLean] 复核结束；结果：失败，%s")
                    % (self._last_invalid_cause)
                )
                
                return False
            
            claim_structure_match = matchs[0]
            fact, prop_field, proof_field, prop_field2, \
                claimed_prop, fact2, claimed_prop2 = claim_structure_match
                
            if fact != fact2 or len(set([self._prop_field, prop_field, prop_field2])) != 1 \
                or claimed_prop != claimed_prop2:
                    
                self._last_invalid_cause = translate(
                    "claim structure（定义关键字 %s 的结构）格式有误！"
                ) % (self._claim_keyword)
                
                if show_progress: print(
                    translate("[CoLean] 复核结束；结果：失败，%s")
                    % (self._last_invalid_cause)
                )
                
                return False
            
            if show_progress: print(translate("[CoLean] 正在逐一复核 claims"))

            keyword_pattern = re.compile(rf'\b{self._claim_keyword}\b')
            keyword_match = list(keyword_pattern.finditer(lean_code))[1:]
            
            if show_progress: keyword_match = tqdm(keyword_match)

            for current_match in keyword_match:
                
                pos = current_match.start()

                tail_text = lean_code[pos:]
                result = self._revalidate_extract_claim_parts(
                    text = tail_text, 
                    start_pos = len(self._claim_keyword),
                )

                if result is None:
                    
                    self._last_invalid_cause = translate(
                        "在位置 %d 发现关键字 %s 后，未能找到符合格式的推理外包逻辑！"
                    ) % (pos, self._claim_keyword)
                    
                    if show_progress: print(
                        translate("[CoLean] 复核结束；结果：失败，%s")
                        % (self._last_invalid_cause)
                    )
                    
                    return False

                prop, verified_facts_raw, revalidator_name = result
                
                prop_pattern = re.compile(
                    r"\{"
                    f"{self._prop_field}"
                    r"\s*:=\s*(.*?)(?=,\s*"
                    f"{proof_field}"
                    r"\s*:=|\})",
                    re.DOTALL
                )
                
                verified_props = prop_pattern.findall(verified_facts_raw)

                if revalidator_name not in self._revalidator_name_to_func:
                    
                    self._last_invalid_cause = translate(
                        "验证器 %s 未知！"
                    ) % revalidator_name
                    
                    if show_progress: print(
                        translate("[CoLean] 复核结束；结果：失败，%s")
                        % (self._last_invalid_cause)
                    )
                    
                    return False

                func = self._revalidator_name_to_func[revalidator_name]

                if not func(prop, verified_props):
                    
                    self._last_invalid_cause = translate(
                        "验证器 %s 复核命题 %s 失败：此验证器不支持命题 %s 导出 %s！"
                    ) % (revalidator_name, prop, ", ".join(verified_props), prop)
                    
                    if show_progress: print(
                        translate("[CoLean] 复核结束；结果：失败，%s")
                        % (self._last_invalid_cause)
                    )
                    
                    return False
                
            if show_progress: print(translate("[CoLean] 复核结束；结果：成功！"))

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
        
