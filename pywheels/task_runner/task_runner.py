from ..i18n import *
from ..typing import *
from ..externals import *
from ..file_tools import *


__all__ = [
    "execute_command",
    "execute_command_async",
    "execute_python_script",
    "execute_python_script_async",
    "run_tasks_concurrently",
    "run_tasks_concurrently_async",
]


def execute_command(
    command: str,
    timeout_seconds: int = 300,
    shell: bool = False,
) -> dict:
    
    def transportable_command_parse(command: str)-> List[str]:
        
        if not command:
            return []
        
        if sys.platform == 'win32':
            return command.split()
        
        else:
            return shlex.split(command)

    result_info = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "timeout": False,
        "exit_code": None,
        "exception": None,
    }

    try:
        if isinstance(command, (list, tuple)):
            args: Union[List[str], str] = command
            
        else:
            args = command if shell else transportable_command_parse(command)

        process = subprocess.run(
            args,
            capture_output = True,
            text = True,
            check = False,
            timeout = timeout_seconds,
            shell = shell,
        )

        result_info["stdout"] = process.stdout
        result_info["stderr"] = process.stderr
        result_info["exit_code"] = process.returncode
        result_info["success"] = (process.returncode == 0)

    except subprocess.TimeoutExpired as e:
        result_info["timeout"] = True
        result_info["exception"] = translate("TimeoutExpired: %s") % (e)

    except Exception as e:
        result_info["exception"] = translate("%s: %s") % (type(e).__name__, e)

    return result_info


async def execute_command_async(
    command: str,
    timeout_seconds: int = 300,
    shell: bool = False,
) -> dict:
    
    def transportable_command_parse(command: str)-> List[str]:
        
        if not command:
            return [] # [FIX 1]
        
        if sys.platform == 'win32':
            return command.split()
        
        else:
            return shlex.split(command)

    result_info = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "timeout": False,
        "exit_code": None,
        "exception": None,
    }
    
    process: Optional[asyncio.subprocess.Process] = None
    try:
        if shell:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            if isinstance(command, (list, tuple)):
                args_tuple: Tuple[str, ...] = tuple(command)
            else:
                args_list: List[str] = transportable_command_parse(command)
                args_tuple = tuple(args_list)
            
            process = await asyncio.create_subprocess_exec(
                *args_tuple,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), 
            timeout=timeout_seconds
        )
        
        result_info["stdout"] = stdout_bytes.decode()
        result_info["stderr"] = stderr_bytes.decode()
        result_info["exit_code"] = process.returncode
        result_info["success"] = (process.returncode == 0)

    except asyncio.TimeoutError as e:
        if process:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
        result_info["timeout"] = True
        result_info["exception"] = translate("TimeoutExpired: %s") % (e)

    except Exception as e:
        if process and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
        result_info["exception"] = translate("%s: %s") % (type(e).__name__, e)

    return result_info


def execute_python_script(
    script_content: str,
    timeout_seconds: int = 300,
    python_command: str = "python",
) -> dict:
    
    temp_file_path = get_temp_file_path(
        suffix=".py",
        prefix = "tmp_TempPythonScript_DeleteMe_",
        directory = None,
    )
        
    with open(
        file = temp_file_path, 
        mode = "w", 
        encoding = "UTF-8",
    ) as temp_file:
        temp_file.write(script_content)
        
    result_info = execute_command(
        command = f"{python_command} {temp_file_path}",
        timeout_seconds = timeout_seconds,
        shell = False,
    )

    delete_file(
        file_path = temp_file_path
    )
    
    return result_info


async def execute_python_script_async(
    script_content: str,
    timeout_seconds: int = 300,
    python_command: str = "python",
) -> dict:
    
    temp_file_path = get_temp_file_path(
        suffix=".py",
        prefix = "tmp_TempPythonScript_DeleteMe_",
        directory = None,
    )
        
    async with aiofiles.open(
        file = temp_file_path, 
        mode = "w", 
        encoding = "UTF-8",
    ) as temp_file:
        await temp_file.write(script_content)
        
    result_info = await execute_command_async(
        command = f"{python_command} {temp_file_path}",
        timeout_seconds = timeout_seconds,
        shell = False,
    )

    await asyncio.to_thread(
        delete_file,
        file_path = temp_file_path
    )
    
    return result_info


_TaskIndexerType = TypeVar("_TaskIndexerType")
_TaskOutputType = TypeVar("_TaskOutputType")

def run_tasks_concurrently(
    task: Callable[..., _TaskOutputType],
    task_indexers: List[_TaskIndexerType],
    task_inputs: List[Tuple[Any, ...]],
    method: Literal["ThreadPoolExecutor", "ProcessPoolExecutor"] = "ThreadPoolExecutor",
    max_workers: Optional[int] = None,
    show_progress_bar: bool = True,
    progress_bar_description: Optional[str] = None,
    local_storage_path: str = "",
    checkpoint_threshold: int = 10,
)-> Dict[_TaskIndexerType, _TaskOutputType]:

    if len(task_indexers) != len(task_inputs):
        raise ValueError(
            translate(
                "task_indexers and task_inputs must have the same length. Got %d indexers and %d inputs."
            ) % (len(task_indexers), len(task_inputs))
        )
    
    if not task_indexers: return {}
    
    executor_class = {
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "ProcessPoolExecutor": ProcessPoolExecutor,
    }[method]
    
    results: Dict[_TaskIndexerType, _TaskOutputType] = {}
    if local_storage_path:
        guarantee_file_exist(file_path=local_storage_path)
        try:
            with open(file=local_storage_path, mode="rb") as file_pointer:
                cached_results = pickle.load(file_pointer)
            results.update(cached_results)
        except (FileNotFoundError, EOFError):
            pass
        
    def _save_cache():
        if not local_storage_path: return
        with open(file=local_storage_path, mode="wb") as file_pointer:
            pickle.dump(results, file_pointer)

    with executor_class(
        max_workers = max_workers
    ) as executor:

        future_to_indexer: Dict[Any, _TaskIndexerType] = {}
        
        for indexer, input_data in zip(task_indexers, task_inputs):
            if indexer in results: continue
            future = executor.submit(task, *input_data)
            future_to_indexer[future] = indexer
            
        future_iterator = as_completed(future_to_indexer)
        
        if show_progress_bar:
            future_iterator = tqdm(
                iterable = future_iterator,
                total = len(task_indexers),
                desc = progress_bar_description,
            )
            future_iterator.update(len(task_indexers) - len(future_to_indexer))

        tick = 0
        for future in future_iterator:
            indexer = future_to_indexer[future]
            try:
                results[indexer] = future.result()
                tick += 1
                if tick % checkpoint_threshold == 0:
                    _save_cache()
            except Exception as error:
                if show_progress_bar and isinstance(future_iterator, tqdm):
                    future_iterator.close()
                raise RuntimeError(
                    translate(
                        "Task failed for indexer '%s': %s"
                    ) % (str(indexer), str(error))
                ) from error
    
    if show_progress_bar and isinstance(future_iterator, tqdm):
        future_iterator.close()

    _save_cache()
    return results


async def run_tasks_concurrently_async(
    task: Callable[..., Coroutine[Any, Any, _TaskOutputType]],
    task_indexers: List[_TaskIndexerType],
    task_inputs: List[Tuple[Any, ...]],
    show_progress_bar: bool = True,
    progress_bar_description: Optional[str] = None,
    local_storage_path: str = "",
    checkpoint_threshold: int = 10,
)-> Dict[_TaskIndexerType, _TaskOutputType]:

    if len(task_indexers) != len(task_inputs):
        raise ValueError(
            translate(
                "task_indexers and task_inputs must have the same length. Got %d indexers and %d inputs."
            ) % (len(task_indexers), len(task_inputs))
        )
    
    if not task_indexers: return {}
    
    results: Dict[_TaskIndexerType, _TaskOutputType] = {}
    if local_storage_path:
        await asyncio.to_thread(guarantee_file_exist, file_path=local_storage_path)
        try:
            async with aiofiles.open(file=local_storage_path, mode="rb") as file_pointer:
                cached_bytes = await file_pointer.read()
                cached_results = pickle.loads(cached_bytes)
            results.update(cached_results)
        except (FileNotFoundError, EOFError):
            pass
        
    async def _save_cache_async():
        if not local_storage_path: return
        async with aiofiles.open(file=local_storage_path, mode="wb") as file_pointer:
            await file_pointer.write(pickle.dumps(results))

    task_to_indexer: Dict[asyncio.Future, _TaskIndexerType] = {}
    tasks_to_run: Set[asyncio.Task] = set()
    
    for indexer, input_data in zip(task_indexers, task_inputs):
        if indexer in results: continue
        coro = task(*input_data)
        task_obj = asyncio.create_task(coro)
        task_to_indexer[task_obj] = indexer
        tasks_to_run.add(task_obj)
        
    if not tasks_to_run:
        return results
    
    task_iterator = asyncio.as_completed(tasks_to_run)
    
    if show_progress_bar:
        task_iterator = tqdm(
            iterable = task_iterator,
            total = len(task_indexers),
            desc = progress_bar_description
        )
        task_iterator.update(len(task_indexers) - len(tasks_to_run))

    tick = 0
    for future in task_iterator:
        indexer = task_to_indexer[future]
        try:
            results[indexer] = await future
            tick += 1
            if tick % checkpoint_threshold == 0:
                await _save_cache_async()
        except Exception as error:
            if show_progress_bar and isinstance(task_iterator, tqdm):
                task_iterator.close()
            raise RuntimeError(
                translate(
                    "Task failed for indexer '%s': %s"
                ) % (str(indexer), str(error))
            ) from error
    
    if show_progress_bar and isinstance(task_iterator, tqdm):
        task_iterator.close()
        
    await _save_cache_async()
    return results