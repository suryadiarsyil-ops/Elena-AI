#!/usr/bin/env python3
"""
ELENA AI - Advanced Version
Enhanced features for Termux & GitHub
"""

import os
import sys
import json
import requests
import readline
import pickle
import hashlib
import base64
import subprocess
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, asdict
from enum import Enum
import random

# ========== COLOR SYSTEM ==========
class Color:
    """Advanced color system for Termux"""
    # Basic colors
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Backgrounds
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    
    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    HIDDEN = "\033[8m"
    
    # Custom gradients
    @staticmethod
    def gradient(text: str, start_color: str, end_color: str) -> str:
        """Create gradient text"""
        colors = [start_color, Color.YELLOW, end_color]
        result = ""
        length = len(text)
        for i, char in enumerate(text):
            color_idx = int((i / length) * (len(colors) - 1))
            result += colors[color_idx] + char
        return result + Color.RESET
    
    @staticmethod
    def rainbow(text: str) -> str:
        """Create rainbow text"""
        colors = [Color.RED, Color.YELLOW, Color.GREEN, Color.CYAN, Color.BLUE, Color.MAGENTA]
        result = ""
        for i, char in enumerate(text):
            result += colors[i % len(colors)] + char
        return result + Color.RESET
    
    @staticmethod
    def progress_bar(percentage: float, width: int = 30) -> str:
        """Create progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        color = Color.GREEN if percentage > 70 else Color.YELLOW if percentage > 30 else Color.RED
        return f"{color}{bar}{Color.RESET} {percentage:.1f}%"

C = Color

# ========== CONFIGURATION ==========
class Config:
    """Configuration manager"""
    
    def __init__(self):
        self.termux_home = Path.home()
        self.config_dir = self.termux_home / ".elena"
        self.data_dir = self.config_dir / "data"
        self.cache_dir = self.config_dir / "cache"
        self.logs_dir = self.config_dir / "logs"
        
        # Files
        self.config_file = self.config_dir / "config.json"
        self.key_file = self.config_dir / "api.key"
        self.history_file = self.data_dir / "history.db"
        self.plugins_dir = self.config_dir / "plugins"
        self.themes_dir = self.config_dir / "themes"
        
        # Create directories
        for dir_path in [self.config_dir, self.data_dir, self.cache_dir, 
                        self.logs_dir, self.plugins_dir, self.themes_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Default config
        self.default_config = {
            "model": "deepseek/deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": True,
            "theme": "default",
            "language": "auto",
            "auto_save": True,
            "history_size": 100,
            "enable_plugins": True,
            "notification": True,
            "log_level": "INFO"
        }
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    config = {**self.default_config, **user_config}
                    return config
            except Exception as e:
                print(f"{C.RED}Error loading config: {e}{C.RESET}")
                return self.default_config.copy()
        return self.default_config.copy()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"{C.RED}Error saving config: {e}{C.RESET}")
            return False
    
    def get(self, key: str, default=None):
        """Get config value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set config value"""
        self.config[key] = value
        return self.save_config()

config = Config()

# ========== MODELS ==========
class AIModel:
    """AI Model manager"""
    
    FREE_MODELS = {
        "deepseek-chat": {
            "id": "deepseek/deepseek-chat",
            "name": "DeepSeek Chat",
            "provider": "DeepSeek",
            "context": 128000,
            "free": True,
            "description": "Model cerdas umum dari DeepSeek"
        },
        "gemini-flash": {
            "id": "google/gemini-2.0-flash-exp:free",
            "name": "Gemini 2.0 Flash",
            "provider": "Google",
            "context": 8192,
            "free": True,
            "description": "Model cepat dari Google Gemini"
        },
        "llama-3.2": {
            "id": "meta-llama/llama-3.2-3b-instruct:free",
            "name": "Llama 3.2 3B",
            "provider": "Meta",
            "context": 8192,
            "free": True,
            "description": "Model ringan dari Meta"
        },
        "mistral-7b": {
            "id": "mistralai/mistral-7b-instruct:free",
            "name": "Mistral 7B",
            "provider": "Mistral AI",
            "context": 32768,
            "free": True,
            "description": "Model kuat dari Mistral AI"
        },
        "qwen-2.5": {
            "id": "qwen/qwen-2.5-32b-instruct:free",
            "name": "Qwen 2.5 32B",
            "provider": "Alibaba",
            "context": 32768,
            "free": True,
            "description": "Model besar dari Alibaba"
        }
    }
    
    PAID_MODELS = {
        "gpt-4": {
            "id": "openai/gpt-4",
            "name": "GPT-4",
            "provider": "OpenAI",
            "context": 8192,
            "free": False,
            "description": "Model paling canggih dari OpenAI"
        },
        "claude-3": {
            "id": "anthropic/claude-3-opus",
            "name": "Claude 3 Opus",
            "provider": "Anthropic",
            "context": 200000,
            "free": False,
            "description": "Model kuat dari Anthropic"
        }
    }
    
    @classmethod
    def get_all_models(cls) -> List[Dict]:
        """Get all available models"""
        all_models = []
        for model_id, info in cls.FREE_MODELS.items():
            all_models.append(info)
        for model_id, info in cls.PAID_MODELS.items():
            all_models.append(info)
        return all_models
    
    @classmethod
    def get_free_models(cls) -> List[Dict]:
        """Get free models only"""
        return list(cls.FREE_MODELS.values())
    
    @classmethod
    def get_model_by_id(cls, model_id: str) -> Optional[Dict]:
        """Get model info by ID"""
        for models in [cls.FREE_MODELS, cls.PAID_MODELS]:
            for info in models.values():
                if info["id"] == model_id:
                    return info
        return None

# ========== CHAT HISTORY ==========
class ChatHistory:
    """Chat history manager with database"""
    
    def __init__(self):
        self.history_file = config.history_file
        self.max_size = config.get("history_size", 100)
        self.sessions = []
        self.current_session = None
        self.load_history()
    
    def load_history(self):
        """Load chat history from database"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'rb') as f:
                    self.sessions = pickle.load(f)
        except Exception:
            self.sessions = []
    
    def save_history(self):
        """Save chat history to database"""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.sessions[-self.max_size:], f)
        except Exception as e:
            print(f"{C.RED}Error saving history: {e}{C.RESET}")
    
    def create_session(self, title: str = None) -> str:
        """Create new chat session"""
        session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        session = {
            "id": session_id,
            "title": title or f"Session {len(self.sessions) + 1}",
            "created": time.time(),
            "messages": [],
            "model": config.get("model"),
            "token_count": 0
        }
        self.sessions.append(session)
        self.current_session = session
        self.save_history()
        return session_id
    
    def add_message(self, role: str, content: str, tokens: int = 0):
        """Add message to current session"""
        if not self.current_session:
            self.create_session()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "tokens": tokens
        }
        
        self.current_session["messages"].append(message)
        self.current_session["token_count"] += tokens
        
        # Update session title if it's the first user message
        if role == "user" and len(self.current_session["messages"]) == 1:
            title = content[:50] + "..." if len(content) > 50 else content
            self.current_session["title"] = title
        
        if config.get("auto_save", True):
            self.save_history()
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        for session in self.sessions:
            if session["id"] == session_id:
                return session
        return None
    
    def list_sessions(self, limit: int = 10) -> List[Dict]:
        """List recent sessions"""
        return self.sessions[-limit:]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        for i, session in enumerate(self.sessions):
            if session["id"] == session_id:
                del self.sessions[i]
                self.save_history()
                return True
        return False
    
    def clear_current(self):
        """Clear current session messages"""
        if self.current_session:
            self.current_session["messages"] = []
            self.current_session["token_count"] = 0
            self.save_history()
    
    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """Export session to different formats"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        if format == "json":
            return json.dumps(session, indent=2, ensure_ascii=False)
        elif format == "txt":
            text = f"Session: {session['title']}\n"
            text += f"Created: {datetime.fromtimestamp(session['created'])}\n"
            text += f"Model: {session['model']}\n"
            text += "=" * 50 + "\n\n"
            
            for msg in session["messages"]:
                role = "👤 User" if msg["role"] == "user" else "🤖 ELENA"
                time_str = datetime.fromtimestamp(msg["timestamp"]).strftime("%H:%M")
                text += f"[{time_str}] {role}:\n{msg['content']}\n\n"
            
            return text
        elif format == "md":
            md = f"# {session['title']}\n\n"
            md += f"**Created**: {datetime.fromtimestamp(session['created'])}\n"
            md += f"**Model**: `{session['model']}`\n\n"
            md += "---\n\n"
            
            for msg in session["messages"]:
                if msg["role"] == "user":
                    md += f"### 👤 User\n\n{msg['content']}\n\n"
                else:
                    md += f"### 🤖 ELENA\n\n{msg['content']}\n\n"
            
            return md
        
        return None

# ========== API CLIENT ==========
class OpenRouterClient:
    """OpenRouter API client with advanced features"""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or self.load_api_key()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ELENA-AI/2.0 (Termux; GitHub)",
            "Accept": "application/json"
        })
    
    def load_api_key(self) -> Optional[str]:
        """Load API key from secure storage"""
        key_file = config.key_file
        if key_file.exists():
            try:
                # Simple encryption/obfuscation
                key_data = key_file.read_text().strip()
                if key_data.startswith("ENC:"):
                    # Decode base64
                    key_data = base64.b64decode(key_data[4:]).decode()
                return key_data
            except Exception:
                return None
        return None
    
    def save_api_key(self, api_key: str, encrypt: bool = True):
        """Save API key securely"""
        try:
            key_data = api_key
            if encrypt:
                key_data = "ENC:" + base64.b64encode(api_key.encode()).decode()
            config.key_file.write_text(key_data)
            # Secure permissions
            if os.name != 'nt':
                os.chmod(config.key_file, 0o600)
            return True
        except Exception as e:
            print(f"{C.RED}Error saving API key: {e}{C.RESET}")
            return False
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test API connection"""
        if not self.api_key:
            return False, "API key tidak ditemukan"
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = self.session.get(
                f"{self.BASE_URL}/auth/key",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, f"Connected as: {data.get('data', {}).get('label', 'Unknown')}"
            else:
                return False, f"API Error: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection failed: {str(e)}"
    
    def chat_completion(self, messages: List[Dict], model: str = None, 
                       temperature: float = None, stream: bool = None) -> Generator:
        """Chat completion with streaming"""
        
        model = model or config.get("model")
        temperature = temperature or config.get("temperature")
        stream = stream or config.get("stream", True)
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": config.get("max_tokens", 2048),
            "stream": stream
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/elena-ai/terminal",
            "X-Title": "ELENA AI Terminal"
        }
        
        try:
            if stream:
                response = self.session.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=60
                )
                
                if response.status_code != 200:
                    error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                    yield {"error": f"API Error {response.status_code}: {error_data}"}
                    return
                
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                delta = data.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                                    yield {"content": content}
                            except json.JSONDecodeError:
                                continue
                
                yield {"done": True, "full_response": full_response}
                
            else:
                # Non-streaming (for summary, etc.)
                response = self.session.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    yield {"content": content}
                    yield {"done": True, "full_response": content}
                else:
                    error_data = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                    yield {"error": f"API Error {response.status_code}: {error_data}"}
                    
        except requests.exceptions.Timeout:
            yield {"error": "⏰ Timeout: Response terlalu lama"}
        except Exception as e:
            yield {"error": f"❌ Error: {str(e)}"}
    
    def get_usage_info(self) -> Optional[Dict]:
        """Get API usage information"""
        if not self.api_key:
            return None
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = self.session.get(
                f"{self.BASE_URL}/usage",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
        except Exception:
            return None
        return None

# ========== PLUGIN SYSTEM ==========
class Plugin:
    """Base plugin class"""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.enabled = True
    
    def on_load(self):
        """Called when plugin is loaded"""
        pass
    
    def on_unload(self):
        """Called when plugin is unloaded"""
        pass
    
    def handle_command(self, command: str, args: List[str]) -> Optional[str]:
        """Handle plugin-specific commands"""
        return None
    
    def process_message(self, message: str) -> Optional[str]:
        """Process incoming messages"""
        return None

class PluginManager:
    """Plugin manager for extensibility"""
    
    def __init__(self):
        self.plugins = {}
        self.load_plugins()
    
    def load_plugins(self):
        """Load all plugins"""
        plugins_dir = config.plugins_dir
        
        # Built-in plugins
        self.register_plugin(CodeExecutorPlugin())
        self.register_plugin(FileManagerPlugin())
        self.register_plugin(WebSearchPlugin())
        self.register_plugin(CalculatorPlugin())
        self.register_plugin(WeatherPlugin())
        
        # Load external plugins
        for plugin_file in plugins_dir.glob("*.py"):
            try:
                # Dynamic import would go here
                pass
            except Exception as e:
                print(f"{C.RED}Error loading plugin {plugin_file}: {e}{C.RESET}")
    
    def register_plugin(self, plugin: Plugin):
        """Register a plugin"""
        self.plugins[plugin.name] = plugin
        plugin.on_load()
    
    def handle_command(self, command: str, args: List[str]) -> Optional[str]:
        """Let plugins handle commands"""
        for plugin in self.plugins.values():
            if plugin.enabled:
                result = plugin.handle_command(command, args)
                if result:
                    return result
        return None
    
    def process_message(self, message: str) -> Optional[str]:
        """Let plugins process messages"""
        for plugin in self.plugins.values():
            if plugin.enabled:
                result = plugin.process_message(message)
                if result:
                    return result
        return None

# ========== BUILT-IN PLUGINS ==========
class CodeExecutorPlugin(Plugin):
    """Code execution plugin"""
    
    def __init__(self):
        super().__init__("code_executor", "1.0")
    
    def handle_command(self, command: str, args: List[str]) -> Optional[str]:
        if command == "/run":
            if not args:
                return f"{C.YELLOW}Usage: /run <language> <code>{C.RESET}"
            
            language = args[0].lower()
            code = " ".join(args[1:])
            
            if language == "python":
                return self.run_python(code)
            elif language == "bash":
                return self.run_bash(code)
            else:
                return f"{C.RED}Language not supported: {language}{C.RESET}"
        
        elif command == "/eval":
            if not args:
                return f"{C.YELLOW}Usage: /eval <python_expression>{C.RESET}"
            
            expression = " ".join(args)
            return self.eval_python(expression)
        
        return None
    
    def run_python(self, code: str) -> str:
        """Execute Python code safely"""
        try:
            # Create safe execution environment
            safe_globals = {
                "__builtins__": {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'range': range,
                    'sum': sum,
                    'min': min,
                    'max': max
                }
            }
            
            # Capture output
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = output = io.StringIO()
            
            exec(code, safe_globals)
            
            sys.stdout = old_stdout
            result = output.getvalue()
            
            return f"{C.GREEN}✅ Output:{C.RESET}\n{result}"
            
        except Exception as e:
            return f"{C.RED}❌ Error: {str(e)}{C.RESET}"
    
    def run_bash(self, code: str) -> str:
        """Execute bash command safely"""
        # Security check
        dangerous = ["rm -rf", "sudo", "chmod 777", "dd", "mkfs", "> /dev/sda"]
        for cmd in dangerous:
            if cmd in code.lower():
                return f"{C.RED}❌ Command not allowed for security{C.RESET}"
        
        try:
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\n{C.RED}Stderr:{C.RESET}\n{result.stderr}"
            
            return f"{C.GREEN}✅ Output:{C.RESET}\n{output}"
            
        except subprocess.TimeoutExpired:
            return f"{C.RED}❌ Timeout: Command took too long{C.RESET}"
        except Exception as e:
            return f"{C.RED}❌ Error: {str(e)}{C.RESET}"
    
    def eval_python(self, expression: str) -> str:
        """Evaluate Python expression safely"""
        try:
            # Very restricted environment
            safe_globals = {"__builtins__": {}}
            safe_locals = {}
            
            result = eval(expression, safe_globals, safe_locals)
            return f"{C.GREEN}✅ Result:{C.RESET} {result}"
            
        except Exception as e:
            return f"{C.RED}❌ Error: {str(e)}{C.RESET}"

class FileManagerPlugin(Plugin):
    """File management plugin"""
    
    def __init__(self):
        super().__init__("file_manager", "1.0")
    
    def handle_command(self, command: str, args: List[str]) -> Optional[str]:
        if command == "/ls" or command == "/dir":
            return self.list_files(args[0] if args else ".")
        
        elif command == "/cat":
            if not args:
                return f"{C.YELLOW}Usage: /cat <filename>{C.RESET}"
            return self.read_file(args[0])
        
        elif command == "/pwd":
            return f"{C.GREEN}Current directory:{C.RESET} {os.getcwd()}"
        
        elif command == "/cd":
            if not args:
                return f"{C.YELLOW}Usage: /cd <directory>{C.RESET}"
            return self.change_dir(args[0])
        
        return None
    
    def list_files(self, path: str) -> str:
        """List files in directory"""
        try:
            if path == "~":
                path = str(Path.home())
            
            files = os.listdir(path)
            
            result = f"{C.BLUE}📁 Directory: {path}{C.RESET}\n\n"
            
            # Group by type
            dirs = []
            py_files = []
            other_files = []
            
            for file in files:
                full_path = os.path.join(path, file)
                if os.path.isdir(full_path):
                    dirs.append(f"{C.BLUE}📁 {file}/{C.RESET}")
                elif file.endswith('.py'):
                    py_files.append(f"{C.GREEN}🐍 {file}{C.RESET}")
                else:
                    other_files.append(f"{C.WHITE}📄 {file}{C.RESET}")
            
            # Display
            if dirs:
                result += "Directories:\n" + "\n".join(dirs) + "\n\n"
            if py_files:
                result += "Python Files:\n" + "\n".join(py_files) + "\n\n"
            if other_files:
                result += "Files:\n" + "\n".join(other_files)
            
            return result
            
        except Exception as e:
            return f"{C.RED}❌ Error: {str(e)}{C.RESET}"
    
    def read_file(self, filename: str) -> str:
        """Read file content"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Limit display
            if len(content) > 10000:
                content = content[:10000] + f"\n{C.YELLOW}...[truncated]{C.RESET}"
            
            return f"{C.GREEN}📖 {filename}:{C.RESET}\n{content}"
            
        except Exception as e:
            return f"{C.RED}❌ Error: {str(e)}{C.RESET}"
    
    def change_dir(self, directory: str) -> str:
        """Change directory"""
        try:
            os.chdir(directory)
            return f"{C.GREEN}✅ Changed to:{C.RESET} {os.getcwd()}"
        except Exception as e:
            return f"{C.RED}❌ Error: {str(e)}{C.RESET}"

class WebSearchPlugin(Plugin):
    """Web search plugin (placeholder)"""
    
    def __init__(self):
        super().__init__("web_search", "1.0")
    
    def handle_command(self, command: str, args: List[str]) -> Optional[str]:
        if command == "/search":
            if not args:
                return f"{C.YELLOW}Usage: /search <query>{C.RESET}"
            
            query = " ".join(args)
            return f"{C.YELLOW}🔍 Search feature coming soon! Query: {query}{C.RESET}"
        
        return None

class CalculatorPlugin(Plugin):
    """Calculator plugin"""
    
    def __init__(self):
        super().__init__("calculator", "1.0")
    
    def handle_command(self, command: str, args: List[str]) -> Optional[str]:
        if command == "/calc":
            if not args:
                return f"{C.YELLOW}Usage: /calc <expression>{C.RESET}"
            
            expression = " ".join(args)
            return self.calculate(expression)
        
        return None
    
    def calculate(self, expression: str) -> str:
        """Calculate mathematical expression"""
        try:
            # Security: only allow safe operations
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return f"{C.RED}❌ Only basic arithmetic allowed{C.RESET}"
            
            result = eval(expression, {"__builtins__": {}}, {})
            return f"{C.GREEN}🧮 Result:{C.RESET} {result}"
            
        except Exception as e:
            return f"{C.RED}❌ Error: {str(e)}{C.RESET}"

class WeatherPlugin(Plugin):
    """Weather plugin (placeholder)"""
    
    def __init__(self):
        super().__init__("weather", "1.0")
    
    def handle_command(self, command: str, args: List[str]) -> Optional[str]:
        if command == "/weather":
            location = " ".join(args) if args else "Jakarta"
            return f"{C.YELLOW}🌤️  Weather feature coming soon for: {location}{C.RESET}"
        
        return None

# ========== ELENA AI CORE ==========
class ElenaAI:
    """Main ELENA AI class"""
    
    def __init__(self):
        self.config = config
        self.client = OpenRouterClient()
        self.history = ChatHistory()
        self.plugins = PluginManager()
        self.current_model = self.config.get("model")
        self.is_running = True
        
        # Initialize
        self.setup_readline()
        self.create_new_session()
    
    def setup_readline(self):
        """Setup readline for better input"""
        try:
            readline.parse_and_bind("tab: complete")
            readline.set_completer(self.completer)
            readline.set_history_length(100)
            
            # Load history file
            history_file = config.data_dir / "input_history"
            if history_file.exists():
                readline.read_history_file(str(history_file))
        except Exception:
            pass
    
    def completer(self, text: str, state: int) -> Optional[str]:
        """Auto-complete for commands"""
        commands = [
            "/help", "/setup", "/models", "/model", "/temp",
            "/clear", "/history", "/save", "/export", "/info",
            "/sessions", "/delete", "/load", "/code", "/tutorial",
            "/run", "/eval", "/ls", "/cat", "/pwd", "/cd",
            "/search", "/calc", "/weather", "/update", "/theme",
            "/plugins", "/usage", "/reset", "/exit"
        ]
        
        options = [cmd for cmd in commands if cmd.startswith(text)]
        if state < len(options):
            return options[state]
        return None
    
    def create_new_session(self, title: str = None):
        """Create new chat session"""
        session_id = self.history.create_session(title)
        print(f"{C.GREEN}🆕 New session created: {session_id}{C.RESET}")
        return session_id
    
    def show_banner(self):
        """Show ELENA AI banner"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        banner = f"""
{C.BRIGHT_CYAN}{C.BOLD}
    ╔═══════════════════════════════════════════════════╗
    ║          {C.RAINBOW}ELENA AI - Advanced Version{C.BRIGHT_CYAN}          ║
    ║      {C.BRIGHT_WHITE}Ethical Learning & Network Assistant{C.BRIGHT_CYAN}      ║
    ║              {C.BRIGHT_YELLOW}Termux & GitHub Ready{C.BRIGHT_CYAN}              ║
    ╚═══════════════════════════════════════════════════╝
{C.RESET}

{C.BRIGHT_GREEN}🚀 Features:{C.RESET}
  • {C.CYAN}Multi-Model AI{C.RESET} (DeepSeek, Gemini, Llama, etc.)
  • {C.CYAN}Plugin System{C.RESET} (Code exec, File manager, etc.)
  • {C.CYAN}Chat History{C.RESET} with export options
  • {C.CYAN}Streaming Responses{C.RESET} in real-time
  • {C.CYAN}Command System{C.RESET} with auto-complete
  • {C.CYAN}Session Management{C.RESET} (save/load/delete)

{C.BRIGHT_YELLOW}📖 Quick Start:{C.RESET}
  1. Type {C.GREEN}/setup{C.RESET} to configure API key
  2. Type {C.GREEN}/help{C.RESET} to see all commands
  3. Type {C.GREEN}/models{C.RESET} to change AI model
  4. Type {C.GREEN}exit{C.RESET} to quit

{C.BRIGHT_MAGENTA}🔗 GitHub: {C.UNDERLINE}https://github.com/elena-ai{C.RESET}
{C.BRIGHT_BLUE}💬 Start chatting or type a command below:{C.RESET}
"""
        print(banner)
    
    def check_api_status(self):
        """Check and display API status"""
        if not self.client.api_key:
            print(f"\n{C.BRIGHT_RED}⚠️  API KEY NOT CONFIGURED{C.RESET}")
            print(f"{C.YELLOW}Use {C.BOLD}/setup{C.RESET}{C.YELLOW} to configure your API key{C.RESET}\n")
            return False
        
        print(f"\n{C.YELLOW}🔍 Checking API connection...{C.RESET}")
        success, message = self.client.test_connection()
        
        if success:
            print(f"{C.GREEN}✅ {message}{C.RESET}")
            
            # Show usage info
            usage = self.client.get_usage_info()
            if usage and 'data' in usage:
                data = usage['data']
                print(f"{C
