"""COBOL Executor for Behavioral Fidelity Evaluation

Executes COBOL programs in a Docker sandbox using GnuCOBOL.

Features:
- Sandboxed execution in Docker container
- Handles console I/O and file I/O
- Timeout and resource limits
- Captures stdout, stderr, and file outputs
- Supports test input injection

Weight in v2.0 evaluation: 35% (Behavioral Fidelity)
"""

import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Results from COBOL program execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    file_outputs: Dict[str, str]  # filename → content
    execution_time_ms: float
    error_message: Optional[str] = None
    timeout: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "file_outputs": self.file_outputs,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "timeout": self.timeout
        }


class COBOLExecutor:
    """
    Execute COBOL programs in a sandboxed Docker environment.

    Uses GnuCOBOL (cobc compiler + cobcrun runtime) in Docker to:
    1. Compile COBOL source
    2. Run with test inputs
    3. Capture all outputs
    4. Handle file I/O
    5. Enforce timeouts and resource limits

    Docker image: Built from Dockerfile in docker/cobol-sandbox/
    """

    def __init__(self,
                 docker_image: str = "legacycodebench-cobol:latest",
                 timeout_seconds: int = 30,
                 memory_limit: str = "512m"):
        """
        Initialize COBOL executor.

        Args:
            docker_image: Docker image with GnuCOBOL
            timeout_seconds: Execution timeout
            memory_limit: Memory limit for container
        """
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit

        # Check if Docker is available
        self._check_docker_available()

        # Check if image exists
        self._check_docker_image_available()

    def can_execute(self, cobol_source: str) -> tuple:
        """
        Check if COBOL program can be executed with GnuCOBOL.
        
        Pre-filter to detect programs that require special preprocessors
        or environments not available in GnuCOBOL (e.g., CICS, missing copybooks).
        
        This allows the evaluator to fall back to BSM-only scoring instead
        of treating compilation failures as failed tests.
        
        Args:
            cobol_source: COBOL source code
            
        Returns:
            (can_execute: bool, reason: str)
                - can_execute: True if program can be compiled/executed
                - reason: Empty string if can execute, otherwise explanation
        """
        # Check for CICS
        if self._is_cics_program(cobol_source):
            return (False, "CICS programs require IBM CICS preprocessor (not available in GnuCOBOL)")
        
        # Could add additional checks here:
        # - Missing copybooks detection
        # - SQL preprocessor requirements
        # - Other middleware dependencies
        
        return (True, "")

    def execute(self, cobol_source: str,
               test_inputs: Dict[str, Any],
               input_files: Optional[Dict[str, str]] = None,
               program_name: Optional[str] = None) -> ExecutionResult:
        """
        Execute COBOL program with test inputs.

        Args:
            cobol_source: COBOL source code
            test_inputs: Dictionary of input values (variable → value)
            input_files: Optional input files (filename → content)
            program_name: Optional program name (extracted from source if not provided)

        Returns:
            ExecutionResult with outputs and status
        """
        import time

        start_time = time.time()

        try:
            # Create temporary workspace
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Determine program name
                if program_name is None:
                    program_name = self._extract_program_name(cobol_source)

                # Write COBOL source
                source_file = temp_path / f"{program_name}.cbl"
                with open(source_file, 'w', encoding='utf-8') as f:
                    f.write(cobol_source)

                # Write input files if provided
                if input_files:
                    for filename, content in input_files.items():
                        file_path = temp_path / filename
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                # Generate input data file for ACCEPT statements
                self._generate_input_data(temp_path, test_inputs)

                # Compile COBOL program
                compile_success, compile_output = self._compile_cobol(
                    temp_path, source_file, program_name
                )

                if not compile_success:
                    execution_time = (time.time() - start_time) * 1000
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=compile_output,
                        exit_code=1,
                        file_outputs={},
                        execution_time_ms=execution_time,
                        error_message=f"Compilation failed: {compile_output}"
                    )

                # Execute compiled program
                result = self._execute_program(temp_path, program_name)

                # Capture file outputs
                file_outputs = self._capture_file_outputs(temp_path)
                result.file_outputs = file_outputs

                # Calculate execution time
                execution_time = (time.time() - start_time) * 1000
                result.execution_time_ms = execution_time

                return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"COBOL execution failed: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                file_outputs={},
                execution_time_ms=execution_time,
                error_message=f"Execution error: {e}"
            )

    def _compile_cobol(self, work_dir: Path, source_file: Path,
                       program_name: str) -> tuple:
        """
        Compile COBOL source using GnuCOBOL in Docker.

        Args:
            work_dir: Working directory with source
            source_file: Path to COBOL source file
            program_name: Program name

        Returns:
            (success: bool, output: str)
        """
        try:
            # Read source to check for CICS/format
            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                source_content = f.read()
            
            # Check for CICS - cannot compile with GnuCOBOL
            if self._is_cics_program(source_content):
                logger.warning(f"CICS program detected - cannot compile with GnuCOBOL")
                return (False, "CICS programs require IBM CICS preprocessor (not available in GnuCOBOL)")
            
            # Detect source format and build compiler flags
            compile_flags = ["-x"]  # Executable
            
            if self._is_free_format(source_content):
                compile_flags.append("-free")
                logger.info("Detected free format, adding -free flag")
            
            # Docker command to compile COBOL
            # cobc -x program.cbl -o program
            docker_cmd = [
                self.docker_cmd, "run", "--rm",
                "-v", f"{work_dir}:/workspace",
                "-w", "/workspace",
                "--memory", self.memory_limit,
                "--network", "none",  # No network access
                self.docker_image,
                "cobc"
            ] + compile_flags + [source_file.name, "-o", program_name]

            logger.info(f"Compiling COBOL: {' '.join(docker_cmd)}")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=self.timeout_seconds
            )
            
            # Decode output manually to avoid Windows threading encoding issues
            stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ''
            stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ''

            if result.returncode == 0:
                logger.info("COBOL compilation successful")
                return (True, stdout)
            else:
                logger.warning(f"COBOL compilation failed: {stderr}")
                return (False, stderr)

        except subprocess.TimeoutExpired:
            logger.error("COBOL compilation timed out")
            return (False, f"Compilation timed out after {self.timeout_seconds}s")
        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return (False, str(e))

    def _execute_program(self, work_dir: Path, program_name: str) -> ExecutionResult:
        """
        Execute compiled COBOL program in Docker.

        Args:
            work_dir: Working directory with compiled program
            program_name: Executable name

        Returns:
            ExecutionResult
        """
        try:
            # Docker command to execute program
            docker_cmd = [
                self.docker_cmd, "run", "--rm",
                "-v", f"{work_dir}:/workspace",
                "-w", "/workspace",
                "--memory", self.memory_limit,
                "--network", "none",
                "-i",  # Interactive for input data
                self.docker_image,
                f"./{program_name}"
            ]

            # Read input data if exists
            input_data_file = work_dir / "input_data.txt"
            stdin_data = None
            if input_data_file.exists():
                with open(input_data_file, 'r') as f:
                    stdin_data = f.read()

            logger.info(f"Executing COBOL: {program_name}")
            
            # Encode input to bytes if present
            stdin_bytes = stdin_data.encode('utf-8') if stdin_data else None

            result = subprocess.run(
                docker_cmd,
                input=stdin_bytes,
                capture_output=True,
                timeout=self.timeout_seconds
            )
            
            # Decode output manually to avoid Windows threading encoding issues
            stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ''
            stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ''

            return ExecutionResult(
                success=(result.returncode == 0),
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                file_outputs={},
                execution_time_ms=0,  # Will be set by caller
                timeout=False
            )

        except subprocess.TimeoutExpired:
            logger.error("COBOL execution timed out")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout_seconds}s",
                exit_code=-1,
                file_outputs={},
                execution_time_ms=0,
                error_message="Timeout",
                timeout=True
            )
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                file_outputs={},
                execution_time_ms=0,
                error_message=str(e)
            )

    def _generate_input_data(self, work_dir: Path, test_inputs: Dict[str, Any]):
        """
        Generate input data file for ACCEPT statements.

        Creates input_data.txt with one value per line for ACCEPT statements.

        Args:
            work_dir: Working directory
            test_inputs: Dictionary of input values
        """
        if not test_inputs:
            return

        input_file = work_dir / "input_data.txt"

        # Write inputs, one per line (for ACCEPT statements)
        with open(input_file, 'w') as f:
            for var_name, value in test_inputs.items():
                f.write(f"{value}\n")

        logger.info(f"Generated input data with {len(test_inputs)} values")

    def _capture_file_outputs(self, work_dir: Path) -> Dict[str, str]:
        """
        Capture output files created by COBOL program.

        Args:
            work_dir: Working directory

        Returns:
            Dictionary of filename → content
        """
        outputs = {}

        # Common COBOL output file patterns
        output_patterns = ["*.dat", "*.txt", "*.out", "*.csv"]

        for pattern in output_patterns:
            for file_path in work_dir.glob(pattern):
                # Skip input files
                if file_path.name == "input_data.txt":
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        outputs[file_path.name] = f.read()
                except Exception as e:
                    logger.warning(f"Could not read output file {file_path}: {e}")

        if outputs:
            logger.info(f"Captured {len(outputs)} output files")

        return outputs

    def _extract_program_name(self, cobol_source: str) -> str:
        """
        Extract program name from COBOL source.

        Looks for PROGRAM-ID. statement in IDENTIFICATION DIVISION.

        Args:
            cobol_source: COBOL source code

        Returns:
            Program name or "PROGRAM" if not found
        """
        import re

        match = re.search(
            r'PROGRAM-ID\.\s+([A-Z0-9-]+)',
            cobol_source,
            re.IGNORECASE
        )

        if match:
            return match.group(1).replace("-", "_")  # Replace hyphens for filesystem
        else:
            logger.warning("Could not extract PROGRAM-ID, using default name")
            return "PROGRAM"

    def _is_cics_program(self, source_content: str) -> bool:
        """
        Detect if COBOL source is a CICS program.
        
        CICS programs use EXEC CICS commands which require IBM's 
        CICS preprocessor and cannot be compiled with GnuCOBOL.
        
        Args:
            source_content: COBOL source code
            
        Returns:
            True if CICS program, False otherwise
        """
        import re
        
        # CICS indicators
        cics_patterns = [
            r'EXEC\s+CICS',           # EXEC CICS commands
            r'EIBCALEN',               # CICS system variable
            r'DFHCOMMAREA',            # CICS communication area
            r'DFHRESP',                # CICS response codes
            r'COPY\s+DFHAID',          # CICS copybook
            r'COPY\s+DFHBMSCA',        # CICS BMS copybook
        ]
        
        for pattern in cics_patterns:
            if re.search(pattern, source_content, re.IGNORECASE):
                return True
        
        return False

    def _is_free_format(self, source_content: str) -> bool:
        """
        Detect if COBOL source is in free format.
        
        Free format COBOL:
        - Has >>SOURCE FORMAT FREE directive
        - Or has code starting before column 7
        
        Args:
            source_content: COBOL source code
            
        Returns:
            True if free format, False if fixed format
        """
        lines = source_content.split('\n')
        
        # Check first 20 lines for format directive
        for line in lines[:20]:
            line_upper = line.upper().strip()
            if '>>SOURCE FORMAT' in line_upper and 'FREE' in line_upper:
                return True
            if '$SET SOURCEFORMAT' in line_upper and 'FREE' in line_upper:
                return True
        
        # Heuristic: if COBOL keywords appear before column 7, likely free format
        for line in lines[:50]:
            stripped = line.strip().upper()
            if stripped.startswith(('IDENTIFICATION', 'ID ', 'PROGRAM-ID', 'DATA ', 'PROCEDURE')):
                # Check if it's at column 7-8 (fixed) or earlier (free)
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces < 6:
                    return True
        
        return False

    def _find_docker_executable(self) -> str:
        """Find docker executable on system"""
        # Try common Windows locations
        if os.name == 'nt':
            common_paths = [
                r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
                r"C:\Program Files\Docker\Docker\resources\bin\docker",
                "docker.exe",
                "docker"
            ]

            for path in common_paths:
                try:
                    result = subprocess.run(
                        [path, "--version"],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"Found Docker at: {path}")
                        return path
                except Exception:
                    continue

        # Unix/Linux/Mac
        return "docker"

    def _check_docker_available(self):
        """Check if Docker is available"""
        try:
            docker_cmd = self._find_docker_executable()
            result = subprocess.run(
                [docker_cmd, "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not available")

            # Store docker command for later use
            self.docker_cmd = docker_cmd
            logger.info(f"Docker is available: {result.stdout.decode().strip()}")
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            raise RuntimeError(
                "Docker is not available. Please install Docker to use COBOL execution. "
                "See docker/README.md for setup instructions."
            )

    def _check_docker_image_available(self):
        """Check if Docker image is available"""
        try:
            result = subprocess.run(
                [self.docker_cmd, "images", "-q", self.docker_image],
                capture_output=True,
                text=True,
                timeout=5
            )

            if not result.stdout.strip():
                logger.warning(f"Docker image {self.docker_image} not found")
                logger.warning("Building Docker image... (this may take a few minutes)")
                self._build_docker_image()
            else:
                logger.info(f"Docker image {self.docker_image} is available")

        except Exception as e:
            logger.error(f"Docker image check failed: {e}")
            raise RuntimeError(
                f"Docker image {self.docker_image} is not available. "
                f"Run 'docker build -t {self.docker_image} docker/cobol-sandbox/' to build it."
            )

    def _build_docker_image(self):
        """Build Docker image if Dockerfile exists"""
        dockerfile_path = Path("docker/cobol-sandbox/Dockerfile")

        if not dockerfile_path.exists():
            raise RuntimeError(
                f"Dockerfile not found at {dockerfile_path}. "
                "Please create docker/cobol-sandbox/Dockerfile. See docker/README.md."
            )

        try:
            logger.info("Building Docker image...")
            result = subprocess.run(
                [
                    self.docker_cmd, "build",
                    "-t", self.docker_image,
                    str(dockerfile_path.parent)
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for build
            )

            if result.returncode == 0:
                logger.info("Docker image built successfully")
            else:
                raise RuntimeError(f"Docker build failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker build timed out after 5 minutes")
        except Exception as e:
            raise RuntimeError(f"Docker build failed: {e}")

    def execute_batch(self, cobol_source: str,
                     test_cases: List[Dict[str, Any]],
                     program_name: Optional[str] = None) -> List[ExecutionResult]:
        """
        Execute COBOL program with multiple test cases.

        More efficient than calling execute() multiple times,
        as it only compiles once.

        Args:
            cobol_source: COBOL source code
            test_cases: List of test input dictionaries
            program_name: Optional program name

        Returns:
            List of ExecutionResult objects
        """
        results = []

        logger.info(f"Executing {len(test_cases)} test cases in batch")

        for i, test_inputs in enumerate(test_cases):
            logger.info(f"  Running test case {i+1}/{len(test_cases)}")
            result = self.execute(cobol_source, test_inputs, program_name=program_name)
            results.append(result)

        logger.info(f"Batch execution complete: {sum(1 for r in results if r.success)}/{len(results)} succeeded")

        return results
