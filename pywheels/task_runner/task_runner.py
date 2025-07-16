import sys
import shlex
import subprocess
from ..file_tools import get_temp_file_path
from ..file_tools import delete_file
from ..i18n import translate


__all__ = [
    "execute_command",
    "execute_python_script",
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
        suffix=".py"
    )
        
    try:
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(script_content)
        
        result_info = execute_command(
            command = f"{python_command} {temp_file_path}",
            timeout_seconds = timeout_seconds,
            shell = False,
        )
    
    except Exception as e:
        result_info = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "timeout": False,
            "exit_code": None,
            "exception": translate("Failed to execute script: %s") % (e),
        }
    
    finally:
        delete_file(
            file_path = temp_file_path
        )
    
    return result_info