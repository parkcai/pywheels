import sys
import shlex
import subprocess
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from ..i18n import translate
from ..file_tools import delete_file
from ..file_tools import get_temp_file_path
from ..typing import *


__all__ = [
    "execute_command",
    "execute_python_script",
    "run_tasks_concurrently",
]


def execute_command(
    command: str,
    timeout_seconds: int = 300,
    shell: bool = False,
) -> dict:
    
    """
    Execute a shell command in a thread-safe environment and capture its output and status.

    This function will:
      - Safely execute shell command using subprocess
      - Capture stdout, stderr and exit code
      - Return a dictionary containing execution result information

    Args:
        command (str): Shell command to execute (as string)
        timeout_seconds (int): Maximum allowed execution time (in seconds). Default 300.
        shell (bool): Whether to enable shell mode. Default False (recommended for security).

    Returns:
        dict: Execution result information containing:
            - success (bool): Whether execution was successful (exit code 0)
            - stdout (str): Command's standard output
            - stderr (str): Command's standard error output
            - timeout (bool): Whether it timed out
            - exit_code (int): Subprocess exit code
            - exception (Optional[str]): Exception type and message if occurred, otherwise None
    """
    
    def transportable_command_parse(command):
        
        if not command:
            return command
        
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
            args = command
            
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


def execute_python_script(
    script_content: str,
    timeout_seconds: int = 300,
    python_command: str = "python",
) -> dict:
    
    """
    Temporarily generate and execute a Python script in thread-safe environment, capturing output and status.

    This function will:
      - Generate unique temp directory and Python script file under thread lock
      - Execute the script, capturing stdout, stderr and exit code
      - Delete temp files/directories to maintain cleanliness
      - Return dictionary containing execution result information

    Args:
        script_content (str): Python script content to execute (as string)
        timeout_seconds (int): Maximum allowed execution time (in seconds). Default 300.
        python_command (str): Python executable command (e.g. "python" or "python3"). Default "python".

    Returns:
        dict: Execution result information containing:
            - success (bool): Whether execution was successful (exit code 0)
            - stdout (str): Script's standard output
            - stderr (str): Script's standard error output
            - timeout (bool): Whether it timed out
            - exit_code (int): Subprocess exit code
            - exception (Optional[str]): Exception type and message if occurred, otherwise None
    """
       
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


TaskIndexerType = TypeVar("TaskIndexerType")
TaskInputType = TypeVar("TaskInputType")
TaskOutputType = TypeVar("TaskOutputType")

def run_tasks_concurrently(
    task: Callable[[TaskInputType], TaskOutputType],
    task_indexers: List[TaskIndexerType],
    task_inputs: List[TaskInputType],
    method: Literal["ThreadPoolExecutor", "ProcessPoolExecutor"] = "ThreadPoolExecutor",
    max_workers: Optional[int] = None,
) -> Dict[TaskIndexerType, TaskOutputType]:
    
    """
    Execute multiple tasks concurrently using thread or process pool, returning indexed results.

    This function will:
      - Validate input parameters for consistency and correctness
      - Create appropriate executor (thread or process pool)
      - Submit all tasks to the executor with proper indexing
      - Collect and map results to their respective indexers
      - Handle exceptions and provide meaningful error information

    Args:
        task: Python callable to execute for each input (accepts single argument, returns output)
        task_indexers: List of unique identifiers for each task (e.g., IDs, names, or keys)
        task_inputs: List of input arguments, each will be passed to the task function
        method: Execution method, either "ThreadPoolExecutor" (default) or "ProcessPoolExecutor"
        max_workers: Maximum number of concurrent workers. None uses default (CPU count for processes)

    Returns:
        Dict[TaskIndexerType, TaskOutputType]: Dictionary mapping each task indexer to its output result

    Raises:
        ValueError: When task_indexers and task_inputs have different lengths
        RuntimeError: When any task execution fails with an exception

    Example:
        >>> def process_item(item: str) -> int:
        ...     return len(item)
        >>> indexers = ["task1", "task2", "task3"]
        >>> inputs = ["hello", "world", "python"]
        >>> results = run_tasks_concurrently(process_item, indexers, inputs)
        >>> print(results)
        {"task1": 5, "task2": 5, "task3": 6}
    """

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
    
    results: Dict[TaskIndexerType, TaskOutputType] = {}
    
    with executor_class(
        max_workers = max_workers
    ) as executor:

        future_to_indexer: Dict[Any, TaskIndexerType] = {}
        
        for indexer, input_data in zip(task_indexers, task_inputs):
            
            future = executor.submit(task, input_data)
            future_to_indexer[future] = indexer

        for future in as_completed(future_to_indexer):
            
            indexer = future_to_indexer[future]
            
            try:
                results[indexer] = future.result()
                
            except Exception as error:

                raise RuntimeError(
                    translate(
                        "Task failed for indexer '%s': %s"
                    ) % (str(indexer), str(error))
                ) from error
    
    return results