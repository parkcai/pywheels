from ..i18n import translate
from ..typing import *
from ..external import *


__all__ = [
    "ModelManager",
    "load_api_keys",
    "get_answer",
]


def _get_answer_online_raw(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: Optional[str],
    images: List[Any],
    temperature: Optional[float],
    top_p: Optional[float],
    max_completion_tokens: Optional[int],
    timeout: Optional[float],
)-> str:

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
    else:
        client = OpenAI(api_key=api_key)
        
    messages = []
    if system_prompt is not None:
        messages.append(
            {"role": "system", "content": system_prompt}
        )
    messages.append(
        {"role": "user", "content": prompt}
    )
    
    optional_params = {}
    if temperature is not None: optional_params["temperature"] = temperature
    if top_p is not None: optional_params["top_p"] = top_p
    if max_completion_tokens is not None: optional_params["max_completion_tokens"] = max_completion_tokens
    if timeout is not None: optional_params["timeout"] = timeout
        
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        stream = False,
        **optional_params,
    )
        
    if isinstance(response, str):
        return response
    else:
        response_content = response.choices[0].message.content
        if response_content is not None:
            return response_content
        else:
            return ""


class ModelManager:
    
    # ----------------------------- Model Manager 初始化 ----------------------------- 
    
    def __init__(self):
        
        self._is_online_model: Dict[str, bool] = {}
        
        self._online_models = {}
        self._online_models_lock: Lock = Lock()
        
    # ----------------------------- 外部动作 ----------------------------- 

    def load_api_keys(
        self, 
        api_keys_path: str,
    )-> None:
        
        if not os.path.exists(api_keys_path) or not os.path.isfile(api_keys_path):
            raise ValueError(
                translate(" api keys 文件 %s 不存在或不是一个文件！")
                % (api_keys_path)
            )
            
        with open(
            file = api_keys_path, 
            mode = 'r',
            encoding = 'UTF-8',
        ) as file:
            api_keys_dict = json.load(file)
        
        with self._online_models_lock:
            for model_name in api_keys_dict:
                self._is_online_model[model_name] = True
                self._online_models[model_name] = {
                    "instances": [
                        {
                            "api_key": api_keys_dict[model_name][index]["api_key"],
                            "base_url": api_keys_dict[model_name][index]["base_url"],
                            "model": api_keys_dict[model_name][index]["model"],
                        }
                        for index in range(len(api_keys_dict[model_name]))
                    ],
                    "next_choice_index": 0,
                }
        
        
    def get_answer(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        images: List[Any] = [],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        trial_num: int = 1,
        trial_interval: int = 5,
        check_and_accept: Callable[[str], bool] = lambda _: True 
    )-> str:
        
        if not self._is_online_model[model]:
            
            raise ValueError(
                translate("模型 %s 未被记录！") % (model)
            )
            
        api_key, base_url, model = self._get_online_model_instance(model)
        
        for trial in range(trial_num):
            try:
                response = _get_answer_online_raw(
                    prompt = prompt,
                    model = model,
                    api_key = api_key,
                    base_url = base_url,
                    system_prompt = system_prompt,
                    images = images,
                    temperature = temperature,
                    top_p = top_p,
                    max_completion_tokens = max_completion_tokens,
                    timeout = timeout,
                )
                if not check_and_accept(response):
                    sleep(
                        max(
                            0, normalvariate(trial_interval, trial_interval / 3)
                        )
                    )
                    continue
                return response
            except Exception as _:
                if trial != trial_num - 1:
                    sleep(
                        max(
                            0, normalvariate(trial_interval, trial_interval / 3)
                        )
                    )
                continue
            
        raise RuntimeError(
            translate("[get_answer 报错] 所有尝试均失败！")
        )
    
    # ----------------------------- 内部动作 ----------------------------- 
  
    def _get_online_model_instance(
        self,
        model_name: str,
    )-> Tuple[str, str, str]:
        
        with self._online_models_lock:
            
            online_model = self._online_models[model_name]
            
            index_backup = online_model["next_choice_index"]
            self._online_models[model_name]["next_choice_index"] = \
                (online_model["next_choice_index"]+1) % len(online_model["instances"])
            
            return (
                online_model["instances"][index_backup]["api_key"],
                online_model["instances"][index_backup]["base_url"],
                online_model["instances"][index_backup]["model"],
            )
    
# ----------------------------- 常用 API -----------------------------

model_manager = ModelManager()


def load_api_keys(
    api_keys_path: str,
)-> None:
    model_manager.load_api_keys(api_keys_path)
     
        
def get_answer(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    images: List[Any] = [],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_completion_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    trial_num: int = 1,
    trial_interval: int = 5,
    check_and_accept: Callable[[str], bool] = lambda _: True 
)-> str:
        
    response = model_manager.get_answer(
        prompt = prompt,
        model = model,
        system_prompt = system_prompt,
        images = images,
        temperature = temperature,
        top_p = top_p,
        max_completion_tokens = max_completion_tokens,
        timeout = timeout,
        trial_num = trial_num,
        trial_interval = trial_interval,
        check_and_accept = check_and_accept,
    )
    
    return response


default_api_keys_path = "api_keys.json"
try:
    load_api_keys(default_api_keys_path)
except Exception as error:
    pass
