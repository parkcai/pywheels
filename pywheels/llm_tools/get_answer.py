from ..i18n import *
from ..typing import *
from ..externals import *


__all__ = [
    "ModelManager",
    "load_api_keys",
    "load_api_keys_async",
    "get_answer",
    "get_answer_async",
    "get_available_models",
    "get_available_models_async",
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
        if os.path.exists(image) and os.path.isfile(image):
            image_type = "file"
        else:
            image_type = "base64"
    elif isinstance(image, bytes):
        image_type = "bytes"
    else:
        raise NotImplementedError(
            translate(
                "暂时无法处理类型为 %s 的图片！"
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
    elif image_type == "base64":
        assert isinstance(image, str)
        image_bytes = base64.b64decode(image)
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


async def _convert_image_to_url_async(
    image: Any,
)-> str:
    
    if isinstance(image, str) and \
        (image.startswith("http://") or image.startswith("https://")):
        image_type = "url"
    elif isinstance(image, str):
        if await aiofiles_os.path.exists(image) and await aiofiles_os.path.isfile(image):
            image_type = "file"
        else:
            image_type = "base64"
    elif isinstance(image, bytes):
        image_type = "bytes"
    else:
        raise NotImplementedError(
            translate(
                "暂时无法处理类型为 %s 的图片！"
            ) % (type(image).__name__)
        )
        
    if image_type == "url":
        assert isinstance(image, str)
        return image
    elif image_type == "file":
        if not await aiofiles_os.path.exists(image) or not await aiofiles_os.path.isfile(image):
            raise FileNotFoundError(
                "图片 %s 被识别为文件，但无法找到非目录文件 %s ！"
            )
        async with aiofiles.open(image, "rb") as file_pointer:
            image_bytes = await file_pointer.read()
        base64_data = base64.b64encode(image_bytes).decode("UTF-8") 
        file_type = _get_file_type_of_image_bytes(image_bytes)
        return f"data:image/{file_type};base64,{base64_data}"
    elif image_type == "base64":
        assert isinstance(image, str)
        image_bytes = base64.b64decode(image)
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


def _parse_tools(
    tools: List[Dict[str, Any]],
)-> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    
    openai_tools_schema: List[Dict[str, Any]] = []
    tool_registry: Dict[str, Callable] = {}
    
    for i, tool_def in enumerate(tools):
        if not isinstance(tool_def, dict):
            raise TypeError(
                translate("tools 列表中的项 (索引 %d) 必须是 dict, 而不是 %s。") 
                % (i, type(tool_def).__name__)
            )
        name = tool_def.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                translate("工具定义 (索引 %d) 缺少 'name' 字段、为空或类型不是 str。") % i
            )
        description = tool_def.get("description")
        if not isinstance(description, str) or not description:
            raise ValueError(
                translate("工具 '%s' (索引 %d) 缺少 'description' 字段、为空或类型不是 str。") 
                % (name, i)
            )
        implementation = tool_def.get("implementation")
        if not callable(implementation):
            raise ValueError(
                translate("工具 '%s' (索引 %d) 缺少 'implementation' 字段或其不是 Callable。") 
                % (name, i)
            )
        tool_registry[name] = implementation
        openai_function_def: Dict[str, Any] = {
            "name": name,
            "description": description
        }
        user_parameters = tool_def.get("parameters")
        if user_parameters is None:
            user_parameters = {}
        if not isinstance(user_parameters, dict):
            raise TypeError(
                translate("工具 '%s' (索引 %d) 的 'parameters' 字段必须是 dict, 而不是 %s。") 
                % (name, i, type(user_parameters).__name__)
            )
        properties: Dict[str, Any] = {}
        required_list: List[str] = []
        for param_name, param_details in user_parameters.items():
            if not isinstance(param_details, dict):
                raise TypeError(
                    translate("工具 '%s' 的参数 '%s' 定义必须是 dict, 而不是 %s。") 
                    % (name, param_name, type(param_details).__name__)
                )
            param_type = param_details.get("type")
            param_description = param_details.get("description")
            is_required = param_details.get("required")
            if not isinstance(param_type, str) or not param_type:
                raise ValueError(
                    translate("工具 '%s' 的参数 '%s' 缺少 'type' 字段、为空或类型不是 str。")
                    % (name, param_name)
                )
            if not isinstance(param_description, str) or not param_description:
                raise ValueError(
                    translate("工具 '%s' 的参数 '%s' 缺少 'description' 字段、为空或类型不是 str。")
                    % (name, param_name)
                )
            if not isinstance(is_required, bool):
                raise TypeError(
                    translate("工具 '%s' 的参数 '%s' 的 'required' 字段必须是 bool (True/False)，不能缺失或为其他类型。")
                    % (name, param_name)
                ) 	
            properties[param_name] = {
                "type": param_type,
                "description": param_description
            }
            if is_required: required_list.append(param_name)
        
        if properties:
            openai_parameters_schema = {
                "type": "object",
                "properties": properties,
                "required": required_list
            }
            openai_function_def["parameters"] = openai_parameters_schema
        openai_tools_schema.append({
            "type": "function",
            "function": openai_function_def
        })
    
    return openai_tools_schema, tool_registry


def _get_answer_raw(
    prompt: Union[str, List[str]],
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
    tools: List[Dict[str, Any]],
    tool_use_trial_num: int,
)-> str:

    if isinstance(prompt, str):
        prompt_list = [prompt]
    elif isinstance(prompt, list):
        prompt_list = prompt
    else:
        raise ValueError(
            translate(
                "prompt 应为 str 或 list，不应为 %s ！"
            ) % (type(prompt).__name__)
        )

    if len(prompt_list) > 1:
        assert len(prompt_list) % 2 == 1, \
            translate(
                "多轮对话的 prompt 列表长度必须是奇数（表示用户 - 助手 - ... - 助手 - 用户）"
            )

    image_placeholder_count = 0
    for text in prompt_list:
        image_placeholder_count += text.count(image_placeholder)
    if image_placeholder_count != len(images):
        raise ValueError(
            translate(
                "prompt 列表共含有 %d 个图片占位符，但提供了 %d 张图片！"
            ) % (image_placeholder_count, len(images))
        )
    
    client_optional_params = {}
    if base_url != "": client_optional_params["base_url"] = base_url
    if timeout is not None: client_optional_params["timeout"] = timeout
    client = OpenAI(
        api_key = api_key,
        **client_optional_params,
    )
    
    messages: List[Any] = []
    if system_prompt is not None:
        messages.append({
            "role": "system", 
            "content": system_prompt
        })
    
    image_index = 0
    for i, text in enumerate(prompt_list):
        role = "user" if i % 2 == 0 else "assistant"
        current_image_placeholder_count = text.count(image_placeholder)
        if role == "assistant" and current_image_placeholder_count > 0:
            raise ValueError(
                translate(
                    "历史消息中, 角色为 'assistant' (助手) 的消息 (索引 %d) "
                    "不允许包含图片占位符！"
                ) % i
            )
        if current_image_placeholder_count == 0:
            messages.append({
                "role": role,
                "content": text,
            })
        else:
            content = []
            seperated_texts = text.split(image_placeholder)
            for j in range(len(seperated_texts)):
                content.append({
                    "type": "text",
                    "text": seperated_texts[j],
                })
                if j == len(seperated_texts) - 1: break
                current_image = images[image_index]
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": _convert_image_to_url(current_image),
                    },
                })
                image_index += 1
            messages.append({
                "role": "user", 
                "content": content,
            })

    optional_params = {}
    if temperature is not None: optional_params["temperature"] = temperature
    if top_p is not None: optional_params["top_p"] = top_p
    if max_completion_tokens is not None: optional_params["max_completion_tokens"] = max_completion_tokens
    
    openai_tools_schema, tool_registry = _parse_tools(tools)
    api_tool_params: Dict[str, Any] = {}
    if openai_tools_schema:
        api_tool_params["tools"] = openai_tools_schema
        api_tool_params["tool_choice"] = "auto"

    for _ in range(tool_use_trial_num):
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            stream = False,
            **optional_params,
            **api_tool_params,
        )
        if isinstance(response, str): return response
        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(response_message)
        if finish_reason == "stop":
            response_content = response_message.content
            return response_content if response_content is not None else ""
        elif finish_reason == "tool_calls":
            assert response_message.tool_calls is not None
            for tool_call in response_message.tool_calls:
                assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments
                if function_name not in tool_registry:
                    raise NameError(
                        translate(
                            "'tool_registry' 中未找到名为 '%s' 的工具！"
                        ) % (function_name)
                    )
                function_to_call = tool_registry[function_name]
                try:
                    function_args = json.loads(function_args_str)
                    function_response = function_to_call(**function_args)
                    if not isinstance(function_response, str):
                        function_response_str = json.dumps(
                            function_response, 
                            ensure_ascii=False,
                        )
                    else:
                        function_response_str = function_response
                except Exception as e:
                    function_response_str = translate(
                        "工具 '%s' 执行失败: %s"
                    ) % (function_name, str(e))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response_str,
                })
            continue
            
        else:
            raise RuntimeError(
                translate(
                    "模型因意外原因停止: %s"
                ) % (finish_reason)
            )

    raise RuntimeError(
        translate(
            "超过最大工具调用次数 (%d)！"
        ) % (tool_use_trial_num)
    )


async def _get_answer_raw_async(
    prompt: Union[str, List[str]],
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
    tools: List[Dict[str, Any]],
    tool_use_trial_num: int,
)-> str:

    if isinstance(prompt, str):
        prompt_list = [prompt]
    elif isinstance(prompt, list):
        prompt_list = prompt
    else:
        raise ValueError(
            translate(
                "prompt 应为 str 或 list，不应为 %s ！"
            ) % (type(prompt).__name__)
        )

    if len(prompt_list) > 1:
        assert len(prompt_list) % 2 == 1, \
            translate(
                "多轮对话的 prompt 列表长度必须是奇数（表示用户 - 助手 - ... - 助手 - 用户）"
            )

    image_placeholder_count = 0
    for text in prompt_list:
        image_placeholder_count += text.count(image_placeholder)
    if image_placeholder_count != len(images):
        raise ValueError(
            translate(
                "prompt 列表共含有 %d 个图片占位符，但提供了 %d 张图片！"
            ) % (image_placeholder_count, len(images))
        )
    
    client_optional_params = {}
    if base_url != "": client_optional_params["base_url"] = base_url
    if timeout is not None: client_optional_params["timeout"] = timeout
    
    client = AsyncOpenAI(
        api_key = api_key,
        **client_optional_params,
    )
    
    messages: List[Any] = []
    if system_prompt is not None:
        messages.append({
            "role": "system", 
            "content": system_prompt
        })
    
    image_index = 0
    for i, text in enumerate(prompt_list):
        role = "user" if i % 2 == 0 else "assistant"
        current_image_placeholder_count = text.count(image_placeholder)
        if role == "assistant" and current_image_placeholder_count > 0:
            raise ValueError(
                translate(
                    "历史消息中, 角色为 'assistant' (助手) 的消息 (索引 %d) "
                    "不允许包含图片占位符！"
                ) % i
            )
        if current_image_placeholder_count == 0:
            messages.append({
                "role": role,
                "content": text,
            })
        else:
            content = []
            seperated_texts = text.split(image_placeholder)
            for j in range(len(seperated_texts)):
                content.append({
                    "type": "text",
                    "text": seperated_texts[j],
                })
                if j == len(seperated_texts) - 1: break
                current_image = images[image_index]
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": await _convert_image_to_url_async(current_image),
                    },
                })
                image_index += 1
            messages.append({
                "role": "user", 
                "content": content,
            })

    optional_params = {}
    if temperature is not None: optional_params["temperature"] = temperature
    if top_p is not None: optional_params["top_p"] = top_p
    if max_completion_tokens is not None: optional_params["max_completion_tokens"] = max_completion_tokens
    
    openai_tools_schema, tool_registry = _parse_tools(tools)
    api_tool_params: Dict[str, Any] = {}
    if openai_tools_schema:
        api_tool_params["tools"] = openai_tools_schema
        api_tool_params["tool_choice"] = "auto"

    for _ in range(tool_use_trial_num):
        response = await client.chat.completions.create(
            model = model,
            messages = messages,
            stream = False,
            **optional_params,
            **api_tool_params,
        )
        if isinstance(response, str): return response
        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(response_message)
        if finish_reason == "stop":
            response_content = response_message.content
            return response_content if response_content is not None else ""
        elif finish_reason == "tool_calls":
            assert response_message.tool_calls is not None
            for tool_call in response_message.tool_calls:
                assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments
                if function_name not in tool_registry:
                    raise NameError(
                        translate(
                            "'tool_registry' 中未找到名为 '%s' 的工具！"
                        ) % (function_name)
                    )
                function_to_call = tool_registry[function_name]
                try:
                    function_args = json.loads(function_args_str)
                    
                    if asyncio.iscoroutinefunction(function_to_call):
                        function_response = await function_to_call(**function_args)
                    else:
                        function_response = await asyncio.to_thread(
                            function_to_call, 
                            **function_args
                        )
                        
                    if not isinstance(function_response, str):
                        function_response_str = json.dumps(
                            function_response, 
                            ensure_ascii=False,
                        )
                    else:
                        function_response_str = function_response
                except Exception as e:
                    function_response_str = translate(
                        "工具 '%s' 执行失败: %s"
                    ) % (function_name, str(e))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response_str,
                })
            continue
            
        else:
            raise RuntimeError(
                translate(
                    "模型因意外原因停止: %s"
                ) % (finish_reason)
            )

    raise RuntimeError(
        translate(
            "超过最大工具调用次数 (%d)！"
        ) % (tool_use_trial_num)
    )


class ModelManager:
    
    def __init__(self):
        
        self._is_online_model: Dict[str, bool] = {}
        
        self._online_models: Dict[str, Any] = {}
        self._online_models_lock: Lock = Lock()
        self._online_models_lock_async: asyncio.Lock = asyncio.Lock()
        
    
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
            self._load_keys_to_memory(api_keys_dict)
    
    
    async def load_api_keys_async(
        self, 
        api_keys_path: str,
    )-> None:
        
        if not await aiofiles_os.path.exists(api_keys_path) or not await aiofiles_os.path.isfile(api_keys_path):
            raise ValueError(
                translate("[get_answer 报错] api keys 文件 %s 不存在或不是一个文件！")
                % (api_keys_path)
            )
            
        async with aiofiles.open(
            file = api_keys_path, 
            mode = 'r',
            encoding = 'UTF-8',
        ) as file:
            content = await file.read()
            api_keys_dict = json.loads(content)
        
        async with self._online_models_lock_async:
            self._load_keys_to_memory(api_keys_dict)


    def get_answer(
        self,
        prompt: Union[str, List[str]],
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
        check_and_accept: Callable[[str], bool] = lambda _: True,
        tools: List[Dict[str, Any]] = [],
        tool_use_trial_num: int = 10,
    )-> str:
        
        if not self._is_online_model[model]:
            raise ValueError(
                translate("[get_answer 报错] 模型 %s 未被记录！") % (model)
            )
            
        api_key, base_url, model = self._get_online_model_instance(model)
        
        last_error = None
        for trial in range(trial_num):
            try:
                response = _get_answer_raw(
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
                    tools = tools,
                    tool_use_trial_num = tool_use_trial_num,
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
        
        
    async def get_answer_async(
        self,
        prompt: Union[str, List[str]],
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
        check_and_accept: Callable[[str], bool] = lambda _: True,
        tools: List[Dict[str, Any]] = [],
        tool_use_trial_num: int = 10,
    )-> str:
        
        if not self._is_online_model[model]:
            raise ValueError(
                translate("[get_answer 报错] 模型 %s 未被记录！") % (model)
            )
            
        api_key, base_url, model = await self._get_online_model_instance_async(model)
        
        last_error = None
        for trial in range(trial_num):
            try:
                response = await _get_answer_raw_async(
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
                    tools = tools,
                    tool_use_trial_num = tool_use_trial_num,
                )
                if not check_and_accept(response):
                    last_error = translate(
                        "模型 %s 的回复未通过 check_and_accept 函数的验收！"
                    ) % (model)
                    await asyncio.sleep(
                        max(
                            0, normalvariate(trial_interval, trial_interval / 3)
                        )
                    )
                    continue
                return response
            except Exception as error:
                last_error = str(error)
                if trial != trial_num - 1:
                    await asyncio.sleep(
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
            
            
    def get_available_models(
        self,
    )-> List[str]:
        
        with self._online_models_lock:
            return [str(model) for model in self._online_models]
    

    async def get_available_models_async(
        self,
    )-> List[str]:
        
        async with self._online_models_lock_async:
            return [str(model) for model in self._online_models]
    

    def _load_keys_to_memory(
        self,
        api_keys_dict: Dict[str, Any],
    )-> None:
        
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

    
    def _get_model_instance_logic(
        self,
        model_name: str,
    )-> Tuple[str, str, str]:
        
        online_model = self._online_models[model_name]
        
        index_backup = online_model["next_choice_index"]
        self._online_models[model_name]["next_choice_index"] = \
            (online_model["next_choice_index"]+1) % len(online_model["instances"])
        
        return (
            online_model["instances"][index_backup]["api_key"],
            online_model["instances"][index_backup]["base_url"],
            online_model["instances"][index_backup]["model"],
        )


    def _get_online_model_instance(
        self,
        model_name: str,
    )-> Tuple[str, str, str]:
        
        with self._online_models_lock:
            return self._get_model_instance_logic(model_name)
    
    
    async def _get_online_model_instance_async(
        self,
        model_name: str,
    )-> Tuple[str, str, str]:
        
        async with self._online_models_lock_async:
            return self._get_model_instance_logic(model_name)
    
# ----------------------------- 常用 API -----------------------------

model_manager = ModelManager()


def load_api_keys(
    api_keys_path: str,
)-> None:
    model_manager.load_api_keys(api_keys_path)
    
    
async def load_api_keys_async(
    api_keys_path: str,
)-> None:
    await model_manager.load_api_keys_async(api_keys_path)
        
        
def get_answer(
    prompt: Union[str, List[str]],
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
    check_and_accept: Callable[[str], bool] = lambda _: True,
    tools: List[Dict[str, Any]] = [],
    tool_use_trial_num: int = 10,
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
        tools = tools,
        tool_use_trial_num = tool_use_trial_num,
    )
    
    return response


async def get_answer_async(
    prompt: Union[str, List[str]],
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
    check_and_accept: Callable[[str], bool] = lambda _: True,
    tools: List[Dict[str, Any]] = [],
    tool_use_trial_num: int = 10,
)-> str:
    
    response = await model_manager.get_answer_async(
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
        tools = tools,
        tool_use_trial_num = tool_use_trial_num,
    )
    
    return response


def get_available_models(
)-> List[str]:
    
    return model_manager.get_available_models()


async def get_available_models_async(
)-> List[str]:
    
    return await model_manager.get_available_models_async()


default_api_keys_path = "api_keys.json"
try:
    load_api_keys(default_api_keys_path)
except Exception as error:
    pass