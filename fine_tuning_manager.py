"""
Fine-tuning Manager for Automatic Model Updates
Handles conversation-based learning and model improvement.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FineTuningManager:
    """Manages automatic fine-tuning from conversation data."""
    
    def __init__(self, conversations_dir: str = "./conversations", 
                 models_dir: str = "./models"):
        self.conversations_dir = conversations_dir
        self.models_dir = models_dir
        self.fine_tune_threshold = 10  # Conversations needed per course
        os.makedirs(conversations_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    
    def check_fine_tuning_readiness(self) -> Dict[str, Dict[str, Any]]:
        """Check which courses are ready for fine-tuning."""
        ready_courses = {}
        
        if not os.path.exists(self.conversations_dir):
            return ready_courses
        
        for filename in os.listdir(self.conversations_dir):
            if filename.endswith('_conversations.jsonl'):
                course_name = filename.replace('_conversations.jsonl', '')
                filepath = os.path.join(self.conversations_dir, filename)
                
                try:
                    conversations = []
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                conversations.append(json.loads(line))
                    
                    conversation_count = len(conversations)
                    fine_tune_triggers = conversation_count // self.fine_tune_threshold
                    
                    if fine_tune_triggers > 0:
                        ready_courses[course_name] = {
                            'conversation_count': conversation_count,
                            'fine_tune_triggers': fine_tune_triggers,
                            'ready_conversations': conversation_count - (conversation_count % self.fine_tune_threshold),
                            'last_conversation': conversations[-1].get('timestamp') if conversations else None,
                            'status': 'ready' if fine_tune_triggers > 0 else 'pending'
                        }
                        
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        return ready_courses
    
    def prepare_training_data(self, course_name: str) -> Optional[List[Dict[str, str]]]:
        """Prepare conversation data for fine-tuning format."""
        filepath = os.path.join(self.conversations_dir, f"{course_name}_conversations.jsonl")
        
        if not os.path.exists(filepath):
            return None
        
        training_data = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        conversation = json.loads(line)
                        
                        # Format for instruction-following fine-tuning
                        training_example = {
                            "instruction": f"Answer this question about {course_name} course content:",
                            "input": conversation.get('question', ''),
                            "output": conversation.get('answer', ''),
                            "course": course_name,
                            "timestamp": conversation.get('timestamp', '')
                        }
                        training_data.append(training_example)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing training data for {course_name}: {e}")
            return None
    
    def save_training_dataset(self, course_name: str, training_data: List[Dict[str, str]]) -> str:
        """Save prepared training data to file."""
        training_file = os.path.join(self.models_dir, f"{course_name}_training_data.jsonl")
        
        try:
            with open(training_file, 'w', encoding='utf-8') as f:
                for example in training_data:
                    f.write(json.dumps(example) + '\n')
            
            logger.info(f"Training data saved: {training_file}")
            return training_file
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            return ""
    
    def create_fine_tuning_config(self, course_name: str, model_type: str = "mistral") -> Dict[str, Any]:
        """Create configuration for fine-tuning process."""
        
        base_models = {
            "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
            "llama": "meta-llama/Llama-2-7b-chat-hf"
        }
        
        config = {
            "base_model": base_models.get(model_type, base_models["mistral"]),
            "course_name": course_name,
            "model_type": model_type,
            "training_config": {
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "batch_size": 4,
                "max_seq_length": 2048,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"] if model_type == "llama" else ["q_proj", "v_proj"]
            },
            "output_dir": os.path.join(self.models_dir, f"{course_name}_{model_type}_finetuned"),
            "created_at": datetime.now().isoformat()
        }
        
        return config
    
    def simulate_fine_tuning(self, course_name: str, model_type: str = "mistral") -> Dict[str, Any]:
        """Simulate fine-tuning process (placeholder for actual implementation)."""
        
        # Check if course is ready
        ready_courses = self.check_fine_tuning_readiness()
        if course_name not in ready_courses:
            return {
                "success": False,
                "error": f"Course {course_name} not ready for fine-tuning",
                "required_conversations": self.fine_tune_threshold,
                "current_conversations": 0
            }
        
        # Prepare training data
        training_data = self.prepare_training_data(course_name)
        if not training_data:
            return {
                "success": False,
                "error": f"Could not prepare training data for {course_name}"
            }
        
        # Save training dataset
        training_file = self.save_training_dataset(course_name, training_data)
        if not training_file:
            return {
                "success": False,
                "error": "Failed to save training dataset"
            }
        
        # Create fine-tuning configuration
        config = self.create_fine_tuning_config(course_name, model_type)
        
        # Save configuration
        config_file = os.path.join(self.models_dir, f"{course_name}_{model_type}_config.json")
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
        
        # Simulate fine-tuning results
        return {
            "success": True,
            "course_name": course_name,
            "model_type": model_type,
            "training_examples": len(training_data),
            "training_file": training_file,
            "config_file": config_file,
            "estimated_time": f"{len(training_data) * 2} minutes",
            "status": "simulated",
            "next_steps": [
                "Training data prepared",
                "Configuration saved", 
                "Ready for actual fine-tuning implementation",
                f"Model will be saved to: {config['output_dir']}"
            ]
        }
    
    def get_fine_tuning_status(self) -> Dict[str, Any]:
        """Get overall fine-tuning status across all courses."""
        ready_courses = self.check_fine_tuning_readiness()
        
        total_ready = len(ready_courses)
        total_conversations = sum(data['conversation_count'] for data in ready_courses.values())
        total_triggers = sum(data['fine_tune_triggers'] for data in ready_courses.values())
        
        return {
            "ready_courses": ready_courses,
            "summary": {
                "total_courses_ready": total_ready,
                "total_conversations": total_conversations,
                "total_fine_tune_triggers": total_triggers,
                "next_course_ready": min(ready_courses.keys()) if ready_courses else None
            },
            "thresholds": {
                "conversations_per_trigger": self.fine_tune_threshold,
                "minimum_for_training": self.fine_tune_threshold
            }
        }
    
    def mark_course_processed(self, course_name: str, processed_count: int):
        """Mark conversations as processed for fine-tuning."""
        # Create a tracking file to avoid reprocessing same conversations
        processed_file = os.path.join(self.models_dir, f"{course_name}_processed.json")
        
        processed_data = {
            "course_name": course_name,
            "processed_conversations": processed_count,
            "last_processed": datetime.now().isoformat()
        }
        
        try:
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2)
            logger.info(f"Marked {processed_count} conversations as processed for {course_name}")
        except Exception as e:
            logger.error(f"Error marking course as processed: {e}")