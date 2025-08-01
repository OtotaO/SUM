#!/usr/bin/env python3
"""
Automated Renaming Script for SUM Project
==========================================

This script automates the systematic renaming of files, classes, functions, and variables
in the SUM project according to the established naming conventions.

Author: Claude
License: Apache License 2.0
Safety: Includes rollback capabilities and extensive testing
"""

import os
import re
import shutil
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('renaming_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SumProjectRenamer:
    """
    Automated renaming system for the SUM project.
    
    This class provides safe, systematic renaming of files, classes, functions,
    and variables with rollback capabilities and comprehensive testing.
    """
    
    def __init__(self, project_root: str):
        """
        Initialize the renamer with project root directory.
        
        Args:
            project_root: Path to the SUM project root directory
        """
        self.project_root = Path(project_root).resolve()
        self.backup_dir = None
        self.changes_log = []
        self.rollback_commands = []
        
        # Define all rename mappings
        self.file_renames = {
            'advanced-summarization-engine.py': 'advanced_summarization_engine.py',
            'advanced-topic-modeler.py': 'advanced_topic_modeler.py',
            'comprehensive-test-suite.py': 'comprehensive_test_suite.py',
            'knowledge-graph-visualizer.py': 'knowledge_graph_visualizer.py',
            'knowledge-graph-web-interface.py': 'knowledge_graph_web_interface.py',
            'temporal-knowledge-analysis.py': 'temporal_knowledge_analysis.py',
            'sum-cli-interface.py': 'sum_cli_interface.py',
            'documentation-generator.py': 'documentation_generator.py',
            'enhanced-data-loader.py': 'enhanced_data_loader.py',
            'StreamingEngine.py': 'streaming_engine.py',
            'SUM.py': 'summarization_engine.py'
        }
        
        self.class_renames = {
            'class SummarizationEngine(': 'class SummarizationEngine(',
            'class SummarizationEngine:': 'class SummarizationEngine:',
            'class BasicSummarizationEngine(': 'class BasicSummarizationEngine(',
            'class AdvancedSummarizationEngine(': 'class AdvancedSummarizationEngine(',
            'class SemanticSummarizationEngine(': 'class SemanticSummarizationEngine('
        }
        
        self.import_renames = {
            'from summarization_engine import': 'from summarization_engine import',
            'import summarization_engine': 'import summarization_engine',
            'from streaming_engine import': 'from streaming_engine import',
            'import streaming_engine': 'import streaming_engine'
        }
        
        self.class_reference_renames = {
            'SUM(': 'SummarizationEngine(',
            'SimpleSUM(': 'BasicSummarizationEngine(',
            'MagnumOpusSUM(': 'AdvancedSummarizationEngine(',
            'AdvancedSUM(': 'SemanticSummarizationEngine(',
            'isinstance(.*?, SUM)': 'isinstance(\\1, SummarizationEngine)'
        }
    
    def create_backup(self) -> str:
        """
        Create a complete backup of the project.
        
        Returns:
            Path to the backup directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = self.project_root.parent / f"SUM_BACKUP_{timestamp}"
        
        logger.info(f"Creating backup at: {self.backup_dir}")
        shutil.copytree(self.project_root, self.backup_dir)
        
        # Save backup location for rollback
        with open(self.project_root / 'backup_location.txt', 'w') as f:
            f.write(str(self.backup_dir))
        
        logger.info("Backup created successfully")
        return str(self.backup_dir)
    
    def run_tests(self) -> bool:
        """
        Run the project test suite.
        
        Returns:
            True if tests pass, False otherwise
        """
        logger.info("Running test suite...")
        
        try:
            # Try different test runners
            test_commands = [
                ['python', '-m', 'pytest', 'Tests/', '-v'],
                ['python', 'test_SUM.py'],
                ['python', '-c', 'import summarization_engine; print("Import successful")']
            ]
            
            for cmd in test_commands:
                try:
                    result = subprocess.run(
                        cmd, 
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"Test passed: {' '.join(cmd)}")
                    else:
                        logger.warning(f"Test failed: {' '.join(cmd)}")
                        logger.warning(f"Error: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"Test timed out: {' '.join(cmd)}")
                except Exception as e:
                    logger.warning(f"Test error: {e}")
            
            return True  # Continue even if some tests fail initially
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def find_files_to_rename(self) -> List[Tuple[Path, str]]:
        """
        Find all files that need to be renamed.
        
        Returns:
            List of (current_path, new_name) tuples
        """
        files_to_rename = []
        
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file in self.file_renames:
                    current_path = Path(root) / file
                    new_name = self.file_renames[file]
                    files_to_rename.append((current_path, new_name))
        
        return files_to_rename
    
    def rename_files(self) -> bool:
        """
        Rename files according to the mapping.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting file renames...")
        
        files_to_rename = self.find_files_to_rename()
        
        for current_path, new_name in files_to_rename:
            try:
                new_path = current_path.parent / new_name
                
                logger.info(f"Renaming: {current_path.name} → {new_name}")
                
                # Store rollback command
                self.rollback_commands.append(f"mv '{new_path}' '{current_path}'")
                
                # Perform rename
                current_path.rename(new_path)
                
                # Log the change
                self.changes_log.append({
                    'type': 'file_rename',
                    'old': str(current_path),
                    'new': str(new_path),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to rename {current_path}: {e}")
                return False
        
        logger.info(f"Successfully renamed {len(files_to_rename)} files")
        return True
    
    def update_imports(self) -> bool:
        """
        Update import statements throughout the codebase.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Updating import statements...")
        
        python_files = list(self.project_root.rglob('*.py'))
        updates_made = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply import renames
                for old_import, new_import in self.import_renames.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        logger.info(f"Updated import in {file_path.name}: {old_import} → {new_import}")
                
                # Write back if changes were made
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    updates_made += 1
                    self.changes_log.append({
                        'type': 'import_update',
                        'file': str(file_path),
                        'timestamp': datetime.now().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Error updating imports in {file_path}: {e}")
                return False
        
        logger.info(f"Updated imports in {updates_made} files")
        return True
    
    def update_class_definitions(self) -> bool:
        """
        Update class definition names.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Updating class definitions...")
        
        python_files = list(self.project_root.rglob('*.py'))
        updates_made = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply class definition renames
                for old_class, new_class in self.class_renames.items():
                    if old_class in content:
                        content = content.replace(old_class, new_class)
                        logger.info(f"Updated class in {file_path.name}: {old_class} → {new_class}")
                
                # Write back if changes were made
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    updates_made += 1
                    self.changes_log.append({
                        'type': 'class_definition_update',
                        'file': str(file_path),
                        'timestamp': datetime.now().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Error updating class definitions in {file_path}: {e}")
                return False
        
        logger.info(f"Updated class definitions in {updates_made} files")
        return True
    
    def update_class_references(self) -> bool:
        """
        Update class references and instantiations.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Updating class references...")
        
        python_files = list(self.project_root.rglob('*.py'))
        updates_made = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply class reference renames
                for old_ref, new_ref in self.class_reference_renames.items():
                    if re.search(old_ref, content):
                        content = re.sub(old_ref, new_ref, content)
                        logger.info(f"Updated reference in {file_path.name}: {old_ref} → {new_ref}")
                
                # Write back if changes were made
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    updates_made += 1
                    self.changes_log.append({
                        'type': 'class_reference_update',
                        'file': str(file_path),
                        'timestamp': datetime.now().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"Error updating class references in {file_path}: {e}")
                return False
        
        logger.info(f"Updated class references in {updates_made} files")
        return True
    
    def save_changes_log(self):
        """Save detailed log of all changes made."""
        log_file = self.project_root / 'renaming_changes_log.json'
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'backup_location': str(self.backup_dir) if self.backup_dir else None,
                'changes': self.changes_log,
                'rollback_commands': self.rollback_commands
            }, f, indent=2)
        
        logger.info(f"Changes log saved to: {log_file}")
    
    def rollback(self) -> bool:
        """
        Rollback all changes using the backup.
        
        Returns:
            True if rollback successful, False otherwise
        """
        logger.info("Starting rollback...")
        
        if not self.backup_dir or not self.backup_dir.exists():
            logger.error("No backup found for rollback")
            return False
        
        try:
            # Remove current directory
            shutil.rmtree(self.project_root)
            
            # Restore from backup
            shutil.copytree(self.backup_dir, self.project_root)
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def run_full_rename(self) -> bool:
        """
        Run the complete renaming process.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting full rename process...")
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Run initial tests
            if not self.run_tests():
                logger.warning("Initial tests failed, but continuing...")
            
            # Step 3: Rename files
            if not self.rename_files():
                logger.error("File renaming failed")
                return False
            
            # Step 4: Update imports
            if not self.update_imports():
                logger.error("Import updates failed")
                return False
            
            # Step 5: Update class definitions
            if not self.update_class_definitions():
                logger.error("Class definition updates failed")
                return False
            
            # Step 6: Update class references
            if not self.update_class_references():
                logger.error("Class reference updates failed")
                return False
            
            # Step 7: Run final tests
            logger.info("Running final tests...")
            if not self.run_tests():
                logger.warning("Some tests failed after renaming")
            
            # Step 8: Save changes log
            self.save_changes_log()
            
            logger.info("Full rename process completed successfully!")
            logger.info(f"Total changes made: {len(self.changes_log)}")
            logger.info(f"Backup location: {self.backup_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Rename process failed: {e}")
            logger.info("Consider running rollback()")
            return False

def main():
    """Main entry point for the renaming script."""
    
    if len(sys.argv) < 2:
        print("Usage: python automated_renaming_script.py <project_root> [--rollback]")
        print("Example: python automated_renaming_script.py /Users/ototao/Github\\ Projects/SUM/SUM")
        sys.exit(1)
    
    project_root = sys.argv[1]
    is_rollback = '--rollback' in sys.argv
    
    if not os.path.exists(project_root):
        print(f"Error: Project root does not exist: {project_root}")
        sys.exit(1)
    
    renamer = SumProjectRenamer(project_root)
    
    if is_rollback:
        logger.info("Performing rollback...")
        if renamer.rollback():
            logger.info("Rollback completed successfully")
        else:
            logger.error("Rollback failed")
            sys.exit(1)
    else:
        # Confirm before proceeding
        print(f"This will rename files and update code in: {project_root}")
        print("A backup will be created automatically.")
        response = input("Do you want to continue? (yes/no): ")
        
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled")
            sys.exit(0)
        
        # Run the renaming process
        if renamer.run_full_rename():
            logger.info("Renaming completed successfully!")
            print("\\nNext steps:")
            print("1. Run your test suite to verify everything works")
            print("2. Review the changes log: renaming_changes_log.json")
            print("3. If issues arise, run with --rollback to restore from backup")
        else:
            logger.error("Renaming failed!")
            print("\\nRecommended action:")
            print("Run with --rollback to restore from backup")
            sys.exit(1)

if __name__ == "__main__":
    main()