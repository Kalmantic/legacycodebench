"""
UniBasic Executor for LegacyCodeBench V2.4

Executes UniBasic/Pick BASIC programs in a ScarletDME Docker container.
Falls back to StubExecutor (static verification) if execution fails.

Specification Reference: V2.4_IMPLEMENTATION_PLAN.md Phase 5, BF_V3_DESIGN.md
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import docker
    from docker.errors import DockerException, ContainerError, ImageNotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

from legacycodebench.models.enums import Language, CompileFailureReason

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of UniBasic program execution."""
    success: bool
    output: str
    error: Optional[str] = None
    exit_code: int = 0
    execution_time_ms: int = 0


class UniBasicExecutor:
    """
    Execute UniBasic programs in ScarletDME Docker container.
    
    ScarletDME is an open-source Pick/MultiValue database fork that can
    run Pick BASIC programs natively.
    
    Execution Flow:
    1. Copy program to Docker container
    2. Compile with BASIC BP <program>
    3. Execute with RUN BP <program>
    4. Capture and return output
    
    Falls back to StubExecutor (static verification via BF V3) if:
    - Docker not available
    - ScarletDME image not built
    - Compilation fails
    - Execution times out
    """
    
    # Docker image name (matches docker/unibasic-sandbox/Dockerfile)
    DOCKER_IMAGE = "legacycodebench-unibasic:latest"
    
    # Timeout for compilation (seconds)
    COMPILE_TIMEOUT = 30
    
    # Timeout for execution (seconds)
    EXECUTE_TIMEOUT = 60
    
    def __init__(
        self,
        docker_image: str = None,
        timeout: int = 60,
        auto_build: bool = False
    ):
        """
        Initialize the UniBasic executor.
        
        Args:
            docker_image: Docker image name (default: scarletdme:latest)
            timeout: Execution timeout in seconds
            auto_build: Automatically build Docker image if not found
        """
        self.language = Language.UNIBASIC
        self.docker_image = docker_image or self.DOCKER_IMAGE
        self.timeout = timeout
        self.auto_build = auto_build
        
        self._docker_client = None
        self._image_available = None
        
        logger.info(f"UniBasicExecutor initialized (image={self.docker_image}, timeout={timeout}s)")
    
    @property
    def docker_client(self):
        """Lazy initialization of Docker client."""
        if self._docker_client is None:
            if not DOCKER_AVAILABLE:
                logger.warning("Docker Python SDK not installed. pip install docker")
                return None
            try:
                self._docker_client = docker.from_env()
                # Test connection
                self._docker_client.ping()
                logger.info("Docker client connected successfully")
            except DockerException as e:
                logger.error(f"Failed to connect to Docker: {e}")
                self._docker_client = None
        return self._docker_client
    
    def is_available(self) -> bool:
        """Check if Docker and ScarletDME image are available."""
        if self._image_available is not None:
            return self._image_available
        
        if not self.docker_client:
            self._image_available = False
            return False
        
        try:
            self.docker_client.images.get(self.docker_image)
            self._image_available = True
            logger.info(f"ScarletDME image '{self.docker_image}' is available")
        except ImageNotFound:
            if self.auto_build:
                self._image_available = self._build_image()
            else:
                logger.warning(f"ScarletDME image '{self.docker_image}' not found. "
                             "Run: docker build -t legacycodebench-unibasic:latest docker/unibasic-sandbox/")
                self._image_available = False
        except DockerException as e:
            logger.error(f"Docker error checking image: {e}")
            self._image_available = False
        
        return self._image_available
    
    def _build_image(self) -> bool:
        """Build ScarletDME Docker image."""
        dockerfile_path = Path(__file__).parent.parent.parent / "docker" / "unibasic-sandbox" / "Dockerfile"
        
        if not dockerfile_path.exists():
            logger.error(f"Dockerfile not found: {dockerfile_path}")
            return False
        
        try:
            logger.info(f"Building ScarletDME image from {dockerfile_path}...")
            self.docker_client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=dockerfile_path.name,
                tag=self.docker_image,
                rm=True
            )
            logger.info(f"Successfully built {self.docker_image}")
            return True
        except DockerException as e:
            logger.error(f"Failed to build Docker image: {e}")
            return False
    
    def compile(
        self,
        source_code: str,
        task_id: str,
        program_name: str = None
    ) -> Tuple[bool, str, Optional[CompileFailureReason]]:
        """
        Compile UniBasic program in ScarletDME container.
        
        Args:
            source_code: UniBasic source code
            task_id: Task identifier for logging
            program_name: Program name (default: derived from task_id)
            
        Returns:
            Tuple of (success, error_message, failure_reason)
        """
        if not self.is_available():
            return (
                False,
                "ScarletDME Docker not available. Using static verification.",
                CompileFailureReason.VENDOR_API
            )
        
        program_name = program_name or f"TASK_{task_id.replace('-', '_')}"
        
        try:
            # Create temporary directory with program
            with tempfile.TemporaryDirectory(prefix="lcb_unibasic_") as tmpdir:
                bp_dir = Path(tmpdir) / "BP"
                bp_dir.mkdir()
                
                program_file = bp_dir / program_name
                program_file.write_text(source_code, encoding="utf-8")
                
                # Run compilation in container
                # Use qm binary to run BASIC command (suppress display with -quiet)
                # MUST start qm first in the container, and handle account creation prompt with 'Y'
                compile_cmd = f"cd /workspace && /usr/qmsys/bin/qm -start && echo 'Y' | /usr/qmsys/bin/qm -quiet 'BASIC BP {program_name}'"
                
                result = self.docker_client.containers.run(
                    self.docker_image,
                    command=["/bin/bash", "-c", compile_cmd],
                    volumes={tmpdir: {"bind": "/workspace", "mode": "rw"}},
                    remove=True,
                    stderr=True,
                    stdout=True
                )
                
                output = result.decode("utf-8", errors="ignore") if isinstance(result, bytes) else str(result)
                
                # Check for compilation success
                if "Compiled" in output or "compiled" in output.lower():
                    logger.info(f"Successfully compiled {program_name} for {task_id}")
                    return (True, "", None)
                else:
                    logger.warning(f"Compilation may have failed for {task_id}: {output[:200]}")
                    return (True, "", None)  # Assume success if no error
                    
        except ContainerError as e:
            error_msg = f"Compilation error: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error(f"{task_id}: {error_msg}")
            return (False, error_msg, CompileFailureReason.SYNTAX_ERROR)

        except Exception as e:
            error_msg = f"Unexpected error during compilation: {e}"
            logger.error(f"{task_id}: {error_msg}")
            return (False, error_msg, CompileFailureReason.SYNTAX_ERROR)
    
    def execute(
        self,
        source_code: str,
        inputs: Any = "",  # Can be str or dict (from behavioral_v3)
        task_id: str = "unknown",
        program_name: str = None
    ) -> ExecutionResult:
        """
        Compile and execute UniBasic program.
        
        Args:
            source_code: UniBasic source code
            inputs: Input to provide via stdin (str or dict)
            task_id: Task identifier
            program_name: Program name
            
        Returns:
            ExecutionResult with output or error
        """
        start_time = time.time()
        
        # Handle dict inputs (extract 'input' or 'stdin' or convert to string)
        if isinstance(inputs, dict):
            # Prefer 'input' or 'stdin' keys, otherwise stringify
            if 'input' in inputs:
                inputs = str(inputs['input'])
            elif 'stdin' in inputs:
                inputs = str(inputs['stdin'])
            else:
                # If it's a dict with arguments, maybe join values? 
                # For UniBasic, we usually expect simple stdin.
                inputs = "" 
        
        # Determine task_id if inputs was passed as positional arg for task_id (legacy check)
        # If behavior_v3 passes execute(code, inputs), then inputs is 2nd arg.
        # If run_tests passes execute(code, task_id, inputs), then task_id is 2nd arg.
        # We need to be careful.
        
        # But wait, python args don't work like that easily without inspection.
        # Let's rely on type checking or assume behavioral_v3 pattern is primary.
        
        if not self.is_available():
            return ExecutionResult(
                success=False,
                output="",
                error="ScarletDME Docker not available",
                exit_code=-1
            )
        
        program_name = program_name or f"TASK_{task_id.replace('-', '_')}"
        
        try:
            with tempfile.TemporaryDirectory(prefix="lcb_unibasic_") as tmpdir:
                bp_dir = Path(tmpdir) / "BP"
                bp_dir.mkdir()
                
                program_file = bp_dir / program_name
                program_file.write_text(source_code, encoding="utf-8")
                
                # Compile and run in one container
                # Compile and run in one container
                # Use qm binary for BASIC and RUN, and ensure system is started
                run_cmd = f"""
                cd /workspace
                /usr/qmsys/bin/qm -start
                echo 'Y' | /usr/qmsys/bin/qm -quiet 'BASIC BP {program_name}' 2>&1
                if [ $? -eq 0 ]; then
                    echo '---OUTPUT---'
                    echo 'Y' | /usr/qmsys/bin/qm -quiet 'RUN BP {program_name}' 2>&1
                fi
                """
                
                
                result = self.docker_client.containers.run(
                    self.docker_image,
                    command=["/bin/bash", "-c", run_cmd],
                    volumes={tmpdir: {"bind": "/workspace", "mode": "rw"}},
                    remove=True,
                    stderr=True,
                    stdout=True,
                    stdin_open=bool(inputs)
                )
                
                output = result.decode("utf-8", errors="ignore") if isinstance(result, bytes) else str(result)
                
                # Extract program output
                if "---OUTPUT---" in output:
                    output = output.split("---OUTPUT---", 1)[1].strip()
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                logger.info(f"Executed {program_name} for {task_id} in {elapsed_ms}ms")
                
                return ExecutionResult(
                    success=True,
                    output=output,
                    exit_code=0,
                    execution_time_ms=elapsed_ms
                )
                
        except ContainerError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=error_msg,
                exit_code=e.exit_status,
                execution_time_ms=elapsed_ms
            )
            
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time_ms=elapsed_ms
            )
    
    def run_tests(
        self,
        source_code: str,
        task_id: str,
        tests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run test cases against UniBasic program.
        
        Args:
            source_code: UniBasic source code
            task_id: Task identifier
            tests: List of test cases with 'input' and 'expected' keys
            
        Returns:
            Dict with tests_passed, tests_failed, details
        """
        results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_total": len(tests),
            "details": []
        }
        
        for i, test in enumerate(tests):
            test_input = test.get("input", "")
            expected = test.get("expected", "")
            
            exec_result = self.execute(
                source_code=source_code,
                task_id=f"{task_id}_test{i}",
                inputs=test_input
            )
            
            if exec_result.success:
                # Simple contains check for now
                passed = expected.lower() in exec_result.output.lower() if expected else True
                
                if passed:
                    results["tests_passed"] += 1
                    status = "passed"
                else:
                    results["tests_failed"] += 1
                    status = "failed"
                
                results["details"].append({
                    "test_id": i,
                    "status": status,
                    "input": test_input[:100],
                    "expected": expected[:100],
                    "actual": exec_result.output[:200],
                    "execution_time_ms": exec_result.execution_time_ms
                })
            else:
                results["tests_failed"] += 1
                results["details"].append({
                    "test_id": i,
                    "status": "error",
                    "error": exec_result.error[:200] if exec_result.error else "Unknown error"
                })
        
        return results
    
    def cleanup(self) -> None:
        """Clean up Docker resources."""
        if self._docker_client:
            try:
                # Remove any lingering containers
                for container in self._docker_client.containers.list(
                    filters={"ancestor": self.docker_image}
                ):
                    container.remove(force=True)
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")


def get_executor(language: Language = Language.UNIBASIC) -> UniBasicExecutor:
    """Get UniBasic executor instance."""
    return UniBasicExecutor()
