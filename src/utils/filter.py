import sys
import re
import logging
from datetime import datetime
from typing import TextIO

# Suppress ALL the noisy loggers
def suppress_all_noise():
    """Suppress all known noisy loggers"""
    noisy_loggers = [
        "autogen_core",
        "httpx", 
        "src.custom_autogen_code.expert",
        "sentence_transformers.SentenceTransformer",
        "chromadb.telemetry.product.posthog",
        "chromadb",
        "src.utils.db_loader",
        "transformers",
        "torch",
        "urllib3",
        "requests"
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.disabled = True
        logger.propagate = False

class OutputFilter:
    def __init__(self, 
                 background_file: str = "full_output.log",
                 clean_messages_file: str = "clean_messages.txt",
                 auto_extract: bool = True):
        
        # Suppress all noise first
        suppress_all_noise()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.background_file = f"src/text_files/{timestamp}_{background_file}"
        self.clean_messages_file = f"src/text_files/{timestamp}_{clean_messages_file}"
        self.auto_extract = auto_extract
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Open background log file
        self.bg_file = open(self.background_file, 'w', encoding='utf-8')
        
        print(f"🔧 Output filtering active:")
        print(f"   📄 Full log: {self.background_file}")
        print(f"   ✨ Clean messages: {self.clean_messages_file}")
        print(f"   🤫 Console noise suppressed")
        print("="*50)
        
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.bg_file.close()
        
        if self.auto_extract:
            # Extract clean messages at the end
            from log_processor import LogProcessor
            messages = LogProcessor.extract_clean_messages(self.background_file, self.clean_messages_file)
            
            # Show summary
            print(f"\n{'='*50}")
            print(f"🎉 Conversation Complete!")
            print(f"   📄 Full log: {self.background_file}")
            print(f"   ✨ Clean messages: {self.clean_messages_file}")
            print(f"   💬 Total messages: {len(messages)}")
            print(f"{'='*50}")
        
    def write(self, text: str):
        # Always save everything to background file
        self.bg_file.write(text)
        self.bg_file.flush()
        
        # Filter what we show on console
        self.filter_and_display(text)
        
        return len(text)
    
    def flush(self):
        self.original_stdout.flush()
        self.bg_file.flush()
        
    def filter_and_display(self, text: str):
        """Simple filtering - just hide noise and show important events"""
        
        # Hide all the noise
        noise_patterns = [
            "INFO:",
            "ERROR:",
            "WARNING:", 
            "DEBUG:",
            "HTTP Request:",
            "Publishing message",
            "Calling message handler",
            "Use pytorch device_name:",
            "Load pretrained SentenceTransformer:",
            "Anonymized telemetry enabled",
            "Failed to send telemetry event",
            "Scanning folder:",
            "Found",
            "File collection complete:",
            "No new files to process!",
            "Files added to memory",
            "---------- TextMessage"  # Hide message headers, we'll extract later
        ]
        
        if any(noise in text for noise in noise_patterns):
            return
            
        # Show important events
        if "Initialized both lobes for Expert" in text:
            agent_match = re.search(r"Expert (\w+)", text)
            if agent_match:
                agent_name = agent_match.group(1).replace('_', ' ').title()
                self.show_simple(f"✅ {agent_name} initialized")
                return
            
        if "Starting internal deliberation" in text:
            agent_match = re.search(r"Expert (\w+)", text)
            if agent_match:
                agent_name = agent_match.group(1).replace('_', ' ').title()
                self.show_simple(f"🤔 {agent_name} thinking...")
                return
                
        if "completed deliberation" in text:
            agent_match = re.search(r"Expert (\w+)", text)
            if agent_match:
                agent_name = agent_match.group(1).replace('_', ' ').title()
                self.show_simple(f"✅ {agent_name} completed analysis")
                return
            
        # Show anything else that might be important
        if text.strip() and len(text.strip()) > 10:
            # Only show if it's not just whitespace or short fragments
            if not any(skip in text.lower() for skip in ['payload', 'delivery_stage', 'sender', 'receiver']):
                self.original_stdout.write(text)
    
    def show_simple(self, message: str):
        """Show simple status messages"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        output = f"[{timestamp}] {message}\n"
        self.original_stdout.write(output)
        self.original_stdout.flush()

# Super simple usage - just wrap your existing code:
def main():
    with OutputFilter("debug.log", "clean_messages.txt"):
        # ALL your existing AutoGen code here - zero changes needed
        print("Starting GHIP Risk Assessment...")
        
        # Your existing imports and setup
        # team = SelectorGroupChatManager(...)
        # result = await team.run(...)
        pass

# Alternative: Manual extraction if you already have a log file
def extract_from_existing_log():
    from log_processor import LogProcessor
    messages = LogProcessor.extract_clean_messages("your_log_file.log", "extracted_messages.txt")
    print(f"Extracted {len(messages)} messages")

# Alternative: Watch log file in real-time (run in separate terminal)
def watch_log_live():
    from log_processor import LogProcessor
    LogProcessor.watch_and_extract("your_log_file.log", "live_messages.txt")

if __name__ == "__main__":
    main()