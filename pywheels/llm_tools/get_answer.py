from ..i18n import translate
from ..typing import *
from ..external import *


__all__ = [
    "ModelManager",
    "load_api_keys",
    "get_answer",
]


def _get_file_type_of_image_bytes(
    image_bytes: bytes,
)-> str:
    
    fallback_file_type = "jpeg"
    if image_bytes.startswith(b'\xFF\xD8\xFF'):
        return "jpeg"
    elif image_bytes.startswith(b'\x89PNG'):
        return "png"
    elif image_bytes.startswith(b'GIF'):
        return "gif"
    else:
        return fallback_file_type


def _convert_image_to_url(
    image: Any,
)-> str:
    
    if isinstance(image, str) and \
        (image.startswith("http://") or image.startswith("https://")):
        image_type = "url"
    elif isinstance(image, str):
        image_type = "file"
    elif isinstance(image, bytes):
        image_type = "bytes"
    else:
        raise NotImplementedError(
            translate(
                "[get_answer 报错] 暂时无法处理类型为 %s 的图片！"
            ) % (type(image).__name__)
        )
        
    if image_type == "url":
        assert isinstance(image, str)
        return image
    elif image_type == "file":
        if not os.path.exists(image) or not os.path.isfile(image):
            raise FileNotFoundError(
                "图片 %s 被识别为文件，但无法找到非目录文件 %s ！"
            )
        with open(image, "rb") as file_pointer:
            image_bytes = file_pointer.read()
        base64_data = base64.b64encode(image_bytes).decode("UTF-8") 
        file_type = _get_file_type_of_image_bytes(image_bytes)
        return f"data:image/{file_type};base64,{base64_data}"
    else:
        assert image_type == "bytes"
        assert isinstance(image, bytes)
        image_bytes = image
        base64_data = base64.b64encode(image_bytes).decode("UTF-8")
        file_type = _get_file_type_of_image_bytes(image_bytes)
        return f"data:image/{file_type};base64,{base64_data}"


def _get_answer_online_raw(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: Optional[str],
    images: List[Any],
    image_placeholder: str,
    temperature: Optional[float],
    top_p: Optional[float],
    max_completion_tokens: Optional[int],
    timeout: Optional[float],
)-> str:
    
    if prompt.count(image_placeholder) != len(images):
        raise ValueError(
            translate(
                "prompt 含有 %d 个图片占位符，但提供了 %d 张图片！"
            ) % (prompt.count(image_placeholder), len(images))
        )
        
    client_optional_params = {}
    if base_url != "": client_optional_params["base_url"] = base_url
    if timeout is not None: client_optional_params["timeout"] = timeout
    client = OpenAI(
        api_key = api_key,
        **client_optional_params,
    )
        
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system", 
            "content": system_prompt
        })
    
    content = []
    seperated_texts = prompt.split(image_placeholder)
    for i in range(len(seperated_texts)):
        content.append({
            "type": "text",
            "text": seperated_texts[i],
        })
        if i == len(seperated_texts) - 1: break
        content.append({
            "type": "image_url",
            "image_url": {
                "url": _convert_image_to_url(images[i]),
            },
        })
    messages.append({
        "role": "user", 
        "content": content,
    })
    
    optional_params = {}
    if temperature is not None: optional_params["temperature"] = temperature
    if top_p is not None: optional_params["top_p"] = top_p
    if max_completion_tokens is not None: optional_params["max_completion_tokens"] = max_completion_tokens
        
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
                translate("[get_answer 报错] api keys 文件 %s 不存在或不是一个文件！")
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
        image_placeholder: str = "<image>",
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
                translate("[get_answer 报错] 模型 %s 未被记录！") % (model)
            )
            
        api_key, base_url, model = self._get_online_model_instance(model)
        
        last_error = None
        for trial in range(trial_num):
            try:
                response = _get_answer_online_raw(
                    prompt = prompt,
                    model = model,
                    api_key = api_key,
                    base_url = base_url,
                    system_prompt = system_prompt,
                    images = images,
                    image_placeholder = image_placeholder,
                    temperature = temperature,
                    top_p = top_p,
                    max_completion_tokens = max_completion_tokens,
                    timeout = timeout,
                )
                if not check_and_accept(response):
                    last_error = translate(
                        "模型 %s 的回复未通过 check_and_accept 函数的验收！"
                    ) % (model)
                    sleep(
                        max(
                            0, normalvariate(trial_interval, trial_interval / 3)
                        )
                    )
                    continue
                return response
            except Exception as error:
                last_error = str(error)
                if trial != trial_num - 1:
                    sleep(
                        max(
                            0, normalvariate(trial_interval, trial_interval / 3)
                        )
                    )
                continue
            
        raise RuntimeError(
            translate(
                "[get_answer 报错] 所有尝试均失败！最后一次尝试的失败原因：%s"
            ) % (last_error)
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
    image_placeholder: str = "<image>",
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
        image_placeholder = image_placeholder,
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
