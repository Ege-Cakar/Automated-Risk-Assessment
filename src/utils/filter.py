import logging
import re
import sys
from datetime import datetime
from threading import Lock
from typing import Dict, Set

class SWIFTStatusFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.lock = Lock()
        self.current_expert = None
        self.current_phase = "Initializing"
        self.experts_seen = set()
        self.tool_calls = 0
        self.llm_calls = 0
        
    def format(self, record):
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            msg = record.getMessage()
            
            # Parse different types of messages
            if self._is_expert_message(record, msg):
                return self._format_expert_message(timestamp, record, msg)
            elif self._is_llm_call(msg):
                return self._format_llm_call(timestamp, msg)
            elif self._is_tool_call(msg):
                return self._format_tool_call(timestamp, msg)
            elif self._is_phase_change(msg):
                return self._format_phase_change(timestamp, msg)
            else:
                return self._format_generic(timestamp, record, msg)
    
    def _is_expert_message(self, record, msg):
        return (record.name.startswith("src.custom_autogen_code.expert") or
                "Expert" in msg or
                "specialist" in msg.lower())
    
    def _is_llm_call(self, msg):
        return "LLMCall" in msg or "HTTP Request: POST https://api.openai.com" in msg
    
    def _is_tool_call(self, msg):
        return "ToolCall" in msg or "tool_name" in msg
    
    def _is_phase_change(self, msg):
        return any(keyword in msg for keyword in [
            "Starting internal deliberation",
            "completed deliberation", 
            "Publishing message",
            "GroupChatStart",
            "GroupChatRequestPublish"
        ])
    
    def _format_expert_message(self, timestamp, record, msg):
        # Extract expert name
        expert_match = re.search(r'(multi_factor_authentication_specialist|data_encryption|identity_proofing|privacy_and_regulatory|application_security|audit_logging|network_and_infrastructure|insider_threat|third_party_integration|summary_agent)', msg)
        
        if expert_match:
            expert = expert_match.group(1).replace('_', ' ').title()
            
            if "Starting internal deliberation" in msg:
                self.current_expert = expert
                self.experts_seen.add(expert)
                return f"🧠 [{timestamp}] {expert} is analyzing..."
            elif "completed deliberation" in msg:
                return f"✅ [{timestamp}] {expert} analysis complete"
            elif "Processing actual content" in msg:
                return f"📝 [{timestamp}] {expert} processing input..."
        
        # Generic expert message
        if "Expert" in msg:
            return f"👨‍💼 [{timestamp}] {msg}"
        
        return f"📋 [{timestamp}] {msg}"
    
    def _format_llm_call(self, timestamp, msg):
        self.llm_calls += 1
        
        if "HTTP Request: POST" in msg:
            return f"🤖 [{timestamp}] AI Model Call #{self.llm_calls}"
        elif "LLMCall" in msg:
            # Try to extract token usage
            token_match = re.search(r'prompt_tokens.*?(\d+).*?completion_tokens.*?(\d+)', msg)
            if token_match:
                prompt_tokens, completion_tokens = token_match.groups()
                return f"🤖 [{timestamp}] AI Response: {prompt_tokens}→{completion_tokens} tokens"
        
        return f"🤖 [{timestamp}] AI Processing..."
    
    def _format_tool_call(self, timestamp, msg):
        self.tool_calls += 1
        
        # Extract tool name
        tool_match = re.search(r'"tool_name":\s*"([^"]+)"', msg)
        if tool_match:
            tool_name = tool_match.group(1)
            return f"🔧 [{timestamp}] Tool: {tool_name}"
        
        return f"🔧 [{timestamp}] Tool Call #{self.tool_calls}"
    
    def _format_phase_change(self, timestamp, msg):
        if "GroupChatStart" in msg:
            self.current_phase = "Discussion Started"
            return f"🎯 [{timestamp}] Phase: {self.current_phase}"
        elif "GroupChatRequestPublish" in msg:
            return f"📤 [{timestamp}] Requesting next speaker..."
        elif "Publishing message" in msg:
            return f"📡 [{timestamp}] Broadcasting message..."
        
        return f"⚡ [{timestamp}] {msg}"
    
    def _format_generic(self, timestamp, record, msg):
        # Different icons for different log levels
        icon = "ℹ️ " if record.levelno == logging.INFO else "⚠️ " if record.levelno == logging.WARNING else "❌ "
        
        # Truncate very long messages
        if len(msg) > 100:
            msg = msg[:97] + "..."
        
        return f"{icon}[{timestamp}] {msg}"

class SWIFTStatusTracker:
    def __init__(self):
        self.current_expert = None
        self.current_phase = "Initializing"
        self.experts_completed = []
        self.start_time = datetime.now()
        
    def print_status_header(self):
        print("=" * 80)
        print("🎯 SWIFT Risk Assessment - Government Medical Records Portal")
        print("=" * 80)
        print()
    
    def print_progress_summary(self):
        elapsed = datetime.now() - self.start_time
        print(f"\n📊 Progress Summary:")
        print(f"   ⏱️  Time Elapsed: {elapsed}")
        print(f"   👥 Current Expert: {self.current_expert or 'None'}")
        print(f"   📋 Phase: {self.current_phase}")
        print(f"   ✅ Experts Completed: {len(self.experts_completed)}")
        print("-" * 50)

def setup_swift_logging():
    """Setup logging for SWIFT assessment with real-time status and formatting"""
    
    # Reduce noise from autogen
    logging.getLogger("autogen_core").setLevel(logging.WARNING)
    logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)  # Keep some HTTP info
    
    # Keep your custom logging at INFO level
    logging.getLogger("src.custom_autogen_code").setLevel(logging.INFO)
    
    # Create custom handler with our formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SWIFTStatusFormatter())
    
    # Clear existing handlers and add our custom one
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # Create status tracker
    status_tracker = SWIFTStatusTracker()
    status_tracker.print_status_header()
    
    return status_tracker

# Enhanced logging with periodic status updates
class PeriodicStatusLogger:
    def __init__(self, interval=30):  # 30 seconds
        self.interval = interval
        self.status_tracker = None
        self.running = False
        self.thread = None
    
    def start(self, status_tracker):
        self.status_tracker = status_tracker
        self.running = True
        from threading import Thread
        self.thread = Thread(target=self._periodic_update)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _periodic_update(self):
        import time
        while self.running:
            time.sleep(self.interval)
            if self.running and self.status_tracker:
                self.status_tracker.print_progress_summary()
