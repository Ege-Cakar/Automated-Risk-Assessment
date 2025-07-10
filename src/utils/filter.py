import logging
import re
import sys
from datetime import datetime
from threading import Lock
from typing import Dict, Set
from autogen_core import EVENT_LOGGER_NAME
import json

class ReadableLogging(logging.Handler):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.file = open(file_path, "a", encoding="utf-8")
        self.request_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.recent_messages = []
        self.cache_size = 50
    
    def emit(self, record):
        try:
            log_msg = self.format(record)
            if '{' not in log_msg:
                return
            
            # Extract JSON from the log message
            json_start = log_msg.find('{')
            json_str = log_msg[json_start:]
            
            try:
                data = json.loads(json_str)
                
                # Process different types of log entries
                if '"type": "LLMCall"' in log_msg:
                    # Extract response from LLMCall
                    if 'response' in data and 'choices' in data['response']:
                        self._process_llm_response(data, record)
                        
                elif '"type": "Message"' in log_msg:
                    # Process message events
                    self._process_message(data, record)
                    
                elif 'agent_response' in data:
                    # Process agent responses
                    self._process_agent_response(data, record)
                else:
                     print(f"[DEBUG] Unknown JSON type: {list(data.keys())[:5]}")
            except json.JSONDecodeError:
                print(f"[DEBUG] Failed to parse JSON from: {json_str[:100]}...")
                pass
        except Exception as e:
            print(f"[DEBUG] Failed to process log message: {e}")
            self.handleError(record)
    def _write_entry(self, timestamp, sender, receiver, content, token_usage=None):
        """Write formatted entry to file"""
        if self._is_duplicate(content, sender, receiver):
            return

        entry = f"\n{'='*80}\n"
        entry += f"[{datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}]\n"
        entry += f"From: {sender}\n"
        entry += f"To: {receiver}\n"
    
        if token_usage:
            entry += f"Tokens: Prompt={token_usage.get('prompt', 0)}, "
            entry += f"Completion={token_usage.get('completion', 0)}, "
            entry += f"Total={token_usage.get('total', 0)}\n"
        
        entry += f"\nContent:\n{content}\n"
        entry += f"{'='*80}\n"
        
        self.file.write(entry)
        self.file.flush()  # Ensure real-time writing
    
    def _process_llm_response(self, data, record):
        """Process LLM response and extract output only"""
        response = data.get('response', {})
        choices = response.get('choices', [])
        usage = response.get('usage', {})
        
        # Update metrics
        self.request_count += 1
        self.total_prompt_tokens += usage.get('prompt_tokens', 0)
        self.total_completion_tokens += usage.get('completion_tokens', 0)
        self.total_tokens += usage.get('total_tokens', 0)
        
        agent_id = data.get('agent_id', 'Unknown')
        # Extract the agent name (before the UUID)
        if '/' in agent_id:
            agent_name = agent_id.split('/')[0]
        else:
            agent_name = agent_id

        # Extract response content
        if choices and len(choices) > 0:
            content = choices[0].get('message', {}).get('content', '')
            if content:
                token_usage = {
                    'prompt': usage.get('prompt_tokens', 0),
                    'completion': usage.get('completion_tokens', 0),
                    'total': usage.get('total_tokens', 0)
                }
                self._write_entry(
                    timestamp=record.created,
                    sender=agent_name,
                    receiver="System",
                    content=content,
                    token_usage=token_usage
                )

    def _process_message(self, data, record):
        """Process message events - only log meaningful content"""
        # Skip delivery stage messages and empty payloads
        delivery_stage = data.get('delivery_stage', '')
        if 'DeliveryStage' in delivery_stage:
            return
        
        # Skip empty or unhelpful payloads
        payload = data.get('payload', '')
        if payload in ['{}', 'Message could not be serialized', '']:
            return
        
        # Only process if there's actual content
        sender = data.get('sender', 'Unknown')
        receiver = data.get('receiver', 'broadcast')
        if receiver is None:
            receiver = 'broadcast'
        
        # Extract sender/receiver names (before the UUID)
        if '/' in sender:
            sender = sender.split('/')[0]
        if '/' in receiver:
            receiver = receiver.split('/')[0]
        
        # Try to extract actual message content from payload
        try:
            if isinstance(payload, str) and payload.startswith('{'):
                payload_data = json.loads(payload)
                # Look for actual message content
                if 'message' in payload_data:
                    msg = payload_data['message']
                    if isinstance(msg, dict) and 'content' in msg:
                        content = msg['content']
                        source = msg.get('source', sender)
                        self._write_entry(
                            timestamp=record.created,
                            sender=source,
                            receiver=receiver,
                            content=content
                        )
                        return
        except:
            pass

    def _is_duplicate(self, content, sender, receiver):
        """Check if this message was recently processed"""
        msg_hash = f"{sender}:{receiver}:{content[:100]}"  # Use first 100 chars
        if msg_hash in self.recent_messages:
            return True
        
        # Add to cache and maintain size
        self.recent_messages.append(msg_hash)
        if len(self.recent_messages) > self.cache_size:
            self.recent_messages.pop(0)
        
        return False

    def _process_agent_response(self, data, record):
        """Process agent response messages"""
        agent_response = data.get('agent_response', {})
        chat_message = agent_response.get('chat_message', {})
        
        source = chat_message.get('source', 'Unknown')
        content = chat_message.get('content', '')
        models_usage = chat_message.get('models_usage', {})
        
        if content:
            # Update token metrics if available
            if models_usage:
                self.total_prompt_tokens += models_usage.get('prompt_tokens', 0)
                self.total_completion_tokens += models_usage.get('completion_tokens', 0)
                
            token_usage = {
                'prompt': models_usage.get('prompt_tokens', 0),
                'completion': models_usage.get('completion_tokens', 0),
                'total': models_usage.get('prompt_tokens', 0) + models_usage.get('completion_tokens', 0)
            } if models_usage else None
            
            self._write_entry(
                timestamp=record.created,
                sender=source,
                receiver="broadcast",
                content=content,
                token_usage=token_usage
            )



def setup_readable_logging():
    # Create your custom handler
    readable_handler = ReadableLogging("readable_logs.txt")
    
    # Get the autogen event logger
    event_logger = logging.getLogger(EVENT_LOGGER_NAME)
    event_logger.addHandler(readable_handler)
    event_logger.setLevel(logging.INFO)
    
    # Also capture autogen_core logs
    autogen_logger = logging.getLogger("autogen_core")
    autogen_logger.addHandler(readable_handler)
    
    # Capture autogen_core.events logs
    events_logger = logging.getLogger("autogen_core.events")
    events_logger.addHandler(readable_handler)
    
    return readable_handler
