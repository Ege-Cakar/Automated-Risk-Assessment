import re
import sys
from io import StringIO
from contextlib import contextmanager

class SimpleMessageFilter:
    """Extract only the actual agent messages from autogen output."""
    
    def __init__(self):
        self.messages = []
        self.current_agent = None
    
    @contextmanager
    def filter_output(self):
        """Context manager to capture and filter output."""
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            yield self
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
            # Process the captured output
            output = captured_output.getvalue()
            self._extract_messages(output)
            self._print_clean_messages()
    
    def _extract_messages(self, output):
        """Extract clean messages from the raw output."""
        lines = output.split('\n')
        
        for line in lines:
            # Look for agent headers like "---------- TextMessage (agent_name) ----------"
            agent_match = re.search(r'---------- TextMessage \(([^)]+)\) ----------', line)
            if agent_match:
                self.current_agent = agent_match.group(1)
                continue
            
            # Skip INFO logs and other noise
            if line.startswith('INFO:') or line.startswith('ERROR:') or line.startswith('WARNING:'):
                continue
            
            # Skip empty lines and autogen internals
            if not line.strip() or 'Publishing message' in line or 'Calling message handler' in line:
                continue
            
            # If we have a current agent and this looks like message content
            if self.current_agent and line.strip():
                # Check if this is the start of a new message block
                if not any(skip_phrase in line for skip_phrase in [
                    'huggingface/tokenizers',
                    'models_usage',
                    'payload',
                    'delivery_stage'
                ]):
                    self.messages.append({
                        'agent': self.current_agent,
                        'content': line.strip()
                    })
    
    def _print_clean_messages(self):
        """Print only the clean conversation."""
        print("\n" + "="*80)
        print("🎯 CLEAN CONVERSATION OUTPUT")
        print("="*80)
        
        current_agent = None
        current_content = []
        
        for msg in self.messages:
            if msg['agent'] != current_agent:
                # Print previous agent's complete message
                if current_agent and current_content:
                    print(f"\n🤖 {current_agent.upper()}:")
                    print("-" * 40)
                    print('\n'.join(current_content))
                
                # Start new agent
                current_agent = msg['agent']
                current_content = [msg['content']]
            else:
                current_content.append(msg['content'])
        
        # Print the last agent's message
        if current_agent and current_content:
            print(f"\n🤖 {current_agent.upper()}:")
            print("-" * 40)
            print('\n'.join(current_content))
        
        print("\n" + "="*80)
