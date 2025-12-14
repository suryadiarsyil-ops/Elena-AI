#!/usr/bin/env python3
"""
ELENA AI - Ethical Learning & Network Assistant
Versi Termux & GitHub Friendly
"""

import os
import sys
import json
import requests
import readline  # Untuk fitur history di Termux
from pathlib import Path
from typing import Generator
import time

# Warna untuk terminal (Termux support)
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    
    # Background colors
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"

C = Colors

# Konfigurasi
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "deepseek/deepseek-chat"  # Default model
DEFAULT_MODELS = [
    "deepseek/deepseek-chat",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-32b-instruct:free"
]

# File penyimpanan di Termux
TERMUX_HOME = Path.home()
CONFIG_DIR = TERMUX_HOME / ".config" / "elena-ai"
KEY_FILE = CONFIG_DIR / "api_key.txt"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "chat_history.json"

# Buat direktori jika belum ada
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Base Persona yang etis dan edukatif
BASE_PERSONA = """Anda adalah ELENA (Ethical Learning & Network Assistant), asisten AI yang berfokus pada pendidikan teknologi dan etika digital.

Prinsip ELENA:
1. **ETIKA**: Selalu bertindak secara etis dan bertanggung jawab
2. **EDUKASI**: Fokus pada pendidikan dan pengetahuan
3. **EMPATI**: Memahami kebutuhan pengguna dengan empati
4. **EFISIENSI**: Memberikan informasi yang jelas dan tepat sasaran
5. **ENABLING**: Memberdayakan pengguna dengan keterampilan praktis

Spesialisasi:
- Pemrograman & Teknologi
- Keamanan Siber (Defensif)
- Machine Learning & AI
- Pengembangan Web & Mobile
- Linux & Terminal commands
- GitHub & Git
- Pendidikan Teknologi

Sikap:
- Ramah dan membantu
- Teknis namun mudah dipahami
- Menghormati privasi dan keamanan
- Menolak permintaan yang tidak etis dengan penjelasan
- Fokus pada solusi yang legal dan konstruktif

Format Respons:
- Gunakan bahasa yang sesuai dengan pengguna
- Berikan contoh kode yang aman dan berkomentar
- Jelaskan konsep dengan analogi jika diperlukan
- Sertakan tips praktis
- Gunakan emoji sesekali untuk keramahan 😊

Ingat: Anda adalah ELENA, asisten yang bertanggung jawab dan edukatif!"""

class ElenaAI:
    def __init__(self):
        self.api_key = None
        self.model = OPENROUTER_MODEL
        self.temperature = 0.7
        self.max_tokens = 2048
        self.conversation_history = []
        self.session_id = int(time.time())
        self.load_config()
        
    def load_config(self):
        """Load konfigurasi dari file"""
        # Load API key
        if KEY_FILE.exists():
            try:
                self.api_key = KEY_FILE.read_text().strip()
            except:
                self.api_key = None
        
        # Load config
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.model = config.get('model', OPENROUTER_MODEL)
                    self.temperature = config.get('temperature', 0.7)
            except:
                pass
    
    def save_config(self):
        """Simpan konfigurasi ke file"""
        config = {
            'model': self.model,
            'temperature': self.temperature,
            'last_updated': time.time()
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except:
            pass
    
    def save_history(self):
        """Simpan riwayat percakapan"""
        if not self.conversation_history:
            return
        
        history_data = {
            'session_id': self.session_id,
            'timestamp': time.time(),
            'model': self.model,
            'messages': self.conversation_history[-20:]  # Simpan 20 pesan terakhir
        }
        
        # Load existing history
        all_history = []
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, 'r') as f:
                    all_history = json.load(f)
                    if not isinstance(all_history, list):
                        all_history = []
            except:
                all_history = []
        
        # Add new session and keep only last 10 sessions
        all_history.append(history_data)
        if len(all_history) > 10:
            all_history = all_history[-10:]
        
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(all_history, f, indent=2)
        except:
            pass
    
    def test_api_key(self, api_key=None):
        """Test API key"""
        test_key = api_key or self.api_key
        if not test_key:
            return False, "API key kosong"
        
        headers = {
            "Authorization": f"Bearer {test_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": "Test: Hello"}],
            "max_tokens": 5
        }
        
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "API key valid"
            elif response.status_code == 401:
                return False, "API key tidak valid"
            elif response.status_code == 429:
                return False, "Rate limit tercapai"
            else:
                return False, f"Error {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Koneksi gagal: {str(e)}"
    
    def setup_api_key(self):
        """Setup API key untuk pertama kali"""
        print(f"\n{C.BLUE}{C.BOLD}🔑 SETUP API KEY ELENA AI{C.RESET}")
        print(f"{C.YELLOW}Langkah 1:{C.RESET} Buka {C.CYAN}https://openrouter.ai/keys{C.RESET}")
        print(f"{C.YELLOW}Langkah 2:{C.RESET} Buat akun (gratis)")
        print(f"{C.YELLOW}Langkah 3:{C.RESET} Buat API key baru")
        print(f"{C.YELLOW}Langkah 4:{C.RESET} Salin key (format: sk-or-v1-...)\n")
        
        while True:
            api_key = input(f"{C.GREEN}Masukkan API key Anda: {C.RESET}").strip()
            
            if not api_key:
                print(f"{C.RED}API key tidak boleh kosong!{C.RESET}")
                continue
            
            if api_key.lower() in ['exit', 'quit']:
                print(f"{C.YELLOW}Setup dibatalkan.{C.RESET}")
                return False
            
            print(f"{C.YELLOW}Memverifikasi API key...{C.RESET}")
            is_valid, message = self.test_api_key(api_key)
            
            if is_valid:
                self.api_key = api_key
                try:
                    KEY_FILE.write_text(api_key)
                    print(f"{C.GREEN}✅ API key berhasil disimpan!{C.RESET}")
                    return True
                except Exception as e:
                    print(f"{C.YELLOW}⚠️  API key valid tetapi gagal disimpan: {e}{C.RESET}")
                    print(f"{C.YELLOW}API key akan digunakan untuk sesi ini saja.{C.RESET}")
                    self.api_key = api_key
                    return True
            else:
                print(f"{C.RED}❌ {message}{C.RESET}")
                print(f"{C.YELLOW}Coba lagi atau ketik 'exit' untuk membatalkan{C.RESET}")
    
    def get_available_models(self):
        """Dapatkan daftar model yang tersedia"""
        return DEFAULT_MODELS
    
    def chat_stream(self, message: str) -> Generator[str, None, None]:
        """Chat dengan streaming response"""
        if not self.api_key:
            yield "❌ Error: API key tidak ditemukan. Gunakan /setup untuk setup API key."
            return
        
        # Tambahkan ke history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Bangun messages untuk API
        messages = [{"role": "system", "content": BASE_PERSONA}]
        messages.extend(self.conversation_history[-10:])  # Ambil 10 pesan terakhir
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/elena-ai",
            "X-Title": "ELENA AI Terminal"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = f"❌ API Error {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    pass
                yield error_msg
                self.conversation_history.append({"role": "assistant", "content": error_msg})
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
                                yield content
                        except json.JSONDecodeError:
                            continue
            
            # Simpan response ke history
            if full_response:
                self.conversation_history.append({"role": "assistant", "content": full_response})
            
        except requests.exceptions.Timeout:
            error_msg = "⏰ Timeout: Response terlalu lama"
            yield error_msg
            self.conversation_history.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            yield error_msg
            self.conversation_history.append({"role": "assistant", "content": error_msg})
    
    def handle_command(self, command: str) -> str:
        """Handle system commands"""
        cmd = command.strip().lower()
        parts = cmd.split()
        
        if cmd == "/help" or cmd == "/?":
            return f"""{C.BOLD}{C.BLUE}📚 ELENA AI - DAFTAR PERINTAH{C.RESET}

{C.GREEN}🧠 AI Commands:{C.RESET}
  /help       - Tampilkan bantuan ini
  /clear      - Hapus riwayat chat
  /history    - Lihat riwayat percakapan
  /save       - Simpan percakapan ke file
  /export     - Export chat sebagai markdown
  
{C.YELLOW}⚙️  System Commands:{C.RESET}
  /setup      - Setup API key
  /models     - Daftar model AI tersedia
  /model <n>  - Ganti model AI (angka)
  /temp <x>   - Set temperature (0.1-1.5)
  /info       - Informasi sistem
  /update     - Update konfigurasi
  
{C.CYAN}🔧 Utility:{C.RESET}
  /code       - Mode bantuan kode
  /explain    - Mode penjelasan detail
  /summarize  - Ringkas percakapan
  /tutorial   - Mode tutorial
  
{C.MAGENTA}📁 File Commands:{C.RESET}
  /read <file> - Baca file teks
  /write       - Buat/edit file
  /list        - List file di direktori
  
{C.WHITE}Type 'exit' or '/exit' to quit{C.RESET}"""
        
        elif cmd == "/clear":
            self.conversation_history.clear()
            return f"{C.GREEN}✅ Riwayat percakapan dihapus{C.RESET}"
        
        elif cmd == "/history":
            if not self.conversation_history:
                return f"{C.YELLOW}📭 Riwayat percakapan kosong{C.RESET}"
            
            history_text = f"{C.BOLD}{C.BLUE}📜 RIWAYAT PERCAKAPAN:{C.RESET}\n"
            for i, msg in enumerate(self.conversation_history[-10:], 1):
                role = "👤 Anda" if msg["role"] == "user" else "🤖 ELENA"
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                history_text += f"\n{C.YELLOW}{i}. {role}:{C.RESET} {content}"
            return history_text
        
        elif cmd == "/info":
            return f"""{C.BOLD}{C.BLUE}🤖 INFORMASI ELENA AI{C.RESET}

{C.GREEN}Version:{C.RESET} 2.0 (Termux/Ready)
{C.GREEN}Model:{C.RESET} {self.model}
{C.GREEN}Temperature:{C.RESET} {self.temperature}
{C.GREEN}Session ID:{C.RESET} {self.session_id}
{C.GREEN}Messages in memory:{C.RESET} {len(self.conversation_history)}
{C.GREEN}API Status:{C.RESET} {'✅ Connected' if self.api_key else '❌ Not connected'}

{C.CYAN}Direktori:{C.RESET}
  Config: {CONFIG_DIR}
  Key: {KEY_FILE}
  History: {HISTORY_FILE}

{C.YELLOW}Github:{C.RESET} https://github.com/elena-ai
{C.MAGENTA}Created for Termux & GitHub!{C.RESET}"""
        
        elif cmd == "/models":
            models = self.get_available_models()
            models_text = f"{C.BOLD}{C.BLUE}🤖 MODEL AI TERSEDIA:{C.RESET}\n"
            for i, model in enumerate(models, 1):
                prefix = "⭐ " if model == self.model else f"{i}. "
                models_text += f"\n{C.YELLOW}{prefix}{C.RESET}{model}"
            models_text += f"\n\n{C.GREEN}Gunakan:{C.RESET} /model <angka>"
            return models_text
        
        elif cmd.startswith("/model ") and len(parts) > 1:
            try:
                model_index = int(parts[1]) - 1
                models = self.get_available_models()
                if 0 <= model_index < len(models):
                    self.model = models[model_index]
                    self.save_config()
                    return f"{C.GREEN}✅ Model diubah ke: {self.model}{C.RESET}"
                else:
                    return f"{C.RED}❌ Nomor model tidak valid. Gunakan /models untuk melihat daftar.{C.RESET}"
            except ValueError:
                return f"{C.RED}❌ Gunakan angka. Contoh: /model 2{C.RESET}"
        
        elif cmd.startswith("/temp ") and len(parts) > 1:
            try:
                temp = float(parts[1])
                if 0.1 <= temp <= 1.5:
                    self.temperature = temp
                    self.save_config()
                    return f"{C.GREEN}✅ Temperature diatur ke: {temp}{C.RESET}"
                else:
                    return f"{C.RED}❌ Temperature harus antara 0.1 dan 1.5{C.RESET}"
            except ValueError:
                return f"{C.RED}❌ Gunakan angka. Contoh: /temp 0.8{C.RESET}"
        
        elif cmd == "/setup":
            if self.setup_api_key():
                return f"{C.GREEN}✅ Setup API key berhasil!{C.RESET}"
            else:
                return f"{C.YELLOW}Setup dibatalkan.{C.RESET}"
        
        elif cmd == "/update":
            self.save_config()
            return f"{C.GREEN}✅ Konfigurasi diperbarui!{C.RESET}"
        
        elif cmd == "/save":
            self.save_history()
            return f"{C.GREEN}✅ Percakapan disimpan ke: {HISTORY_FILE}{C.RESET}"
        
        elif cmd == "/code":
            return f"""{C.GREEN}💻 MODE BANTUAN KODE{C.RESET}

{C.YELLOW}ELENA sekarang dalam mode bantuan pemrograman!{C.RESET}
Saya akan fokus membantu dengan:
• Debugging kode
• Optimasi algoritma
• Best practices
• Penjelasan konsep
• Contoh implementasi

{C.CYAN}Contoh pertanyaan:{C.RESET}
• "Bantu debug fungsi Python ini"
• "Bagaimana cara optimasi query SQL?"
• "Buatkan contoh REST API dengan FastAPI"
• "Jelaskan konsep async/await"

{C.MAGENTA}Ayo tanyakan masalah kode Anda! 😊{C.RESET}"""
        
        elif cmd == "/tutorial":
            return f"""{C.GREEN}🎓 MODE TUTORIAL{C.RESET}

{C.YELLOW}ELENA sekarang dalam mode tutorial!{C.RESET}
Saya akan memberikan panduan langkah-demi-langkah untuk:
• Belajar pemrograman dari dasar
• Setup environment development
• Tutorial framework/library
• Proyek praktis
• Tips dan trik

{C.CYAN}Pilih topik:{C.RESET}
1. Python untuk pemula
2. Web Development (HTML/CSS/JS)
3. Machine Learning dasar
4. GitHub & Git
5. Linux/Termux basics

{C.MAGENTA}Tanyakan "tutorial [topik]" untuk mulai! ✨{C.RESET}"""
        
        elif cmd == "/list":
            try:
                files = os.listdir('.')
                files_text = f"{C.BOLD}{C.BLUE}📁 FILES DI DIREKTORI INI:{C.RESET}\n"
                for file in files[:20]:  # Tampilkan 20 file pertama
                    if os.path.isdir(file):
                        files_text += f"{C.BLUE}📁 {file}/{C.RESET}\n"
                    else:
                        files_text += f"{C.GREEN}📄 {file}{C.RESET}\n"
                if len(files) > 20:
                    files_text += f"\n{C.YELLOW}... dan {len(files)-20} file lainnya{C.RESET}"
                return files_text
            except Exception as e:
                return f"{C.RED}❌ Error: {str(e)}{C.RESET}"
        
        else:
            return f"{C.RED}❌ Perintah tidak dikenali. Ketik {C.YELLOW}/help{C.RED} untuk bantuan.{C.RESET}"


def print_banner():
    """Print banner ELENA AI"""
    banner = f"""
{C.BLUE}{C.BOLD}
    ███████╗██╗     ███████╗███╗   ██╗ █████╗
    ██╔════╝██║     ██╔════╝████╗  ██║██╔══██╗
    █████╗  ██║     █████╗  ██╔██╗ ██║███████║
    ██╔══╝  ██║     ██╔══╝  ██║╚██╗██║██╔══██║
    ███████╗███████╗███████╗██║ ╚████║██║  ██║
    ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝
    
    {C.CYAN}Ethical Learning & Network Assistant{C.RESET}
    {C.GREEN}Version 2.0 - Termux & GitHub Ready{C.RESET}
    {C.YELLOW}Type '/help' for commands, 'exit' to quit{C.RESET}
{C.RESET}
"""
    print(banner)


def main():
    """Main function"""
    # Setup readline untuk history di Termux
    try:
        readline.parse_and_bind("tab: complete")
        readline.set_history_length(100)
    except:
        pass
    
    # Clear screen untuk Termux
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print_banner()
    
    # Inisialisasi ELENA AI
    elena = ElenaAI()
    
    # Check API key
    if not elena.api_key:
        print(f"{C.YELLOW}⚠️  API key tidak ditemukan!{C.RESET}")
        print(f"{C.CYAN}Gunakan perintah {C.BOLD}/setup{C.RESET}{C.CYAN} untuk setup API key{C.RESET}\n")
    else:
        # Test API key
        print(f"{C.YELLOW}🔍 Memeriksa API key...{C.RESET}")
        is_valid, message = elena.test_api_key()
        if is_valid:
            print(f"{C.GREEN}✅ {message}{C.RESET}\n")
        else:
            print(f"{C.RED}❌ {message}{C.RESET}")
            print(f"{C.YELLOW}Gunakan {C.BOLD}/setup{C.RESET}{C.YELLOW} untuk memperbarui API key{C.RESET}\n")
    
    # Prompt awal
    print(f"{C.MAGENTA}💬 Mulai chat dengan ELENA atau ketik '/help' untuk bantuan{C.RESET}\n")
    
    # Conversation loop
    while True:
        try:
            # Input dengan prompt menarik
            prompt = f"{C.GREEN}👤 Anda{C.RESET} {C.BLUE}➜{C.RESET} "
            try:
                user_input = input(prompt).strip()
            except EOFError:
                print(f"\n{C.YELLOW}👋 Sampai jumpa!{C.RESET}")
                break
            except KeyboardInterrupt:
                print(f"\n{C.YELLOW}🚪 Keluar? Ketik 'exit' untuk keluar{C.RESET}")
                continue
            
            # Handle exit
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"\n{C.GREEN}🤖 ELENA: Sampai jumpa! Terima kasih telah menggunakan ELENA AI! ✨{C.RESET}")
                elena.save_history()
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                response = elena.handle_command(user_input)
                print(f"\n{C.CYAN}⚙️  System:{C.RESET} {response}\n")
                continue
            
            # Chat dengan ELENA
            print(f"\n{C.MAGENTA}🤖 ELENA{C.RESET} {C.BLUE}➜{C.RESET} ", end='', flush=True)
            
            # Streaming response
            full_response = ""
            for chunk in elena.chat_stream(user_input):
                print(chunk, end='', flush=True)
                full_response += chunk
            
            print("\n")  # New line after response
            
        except Exception as e:
            print(f"\n{C.RED}⚠️  Error: {str(e)}{C.RESET}")
            print(f"{C.YELLOW}Melanjutkan...{C.RESET}\n")


def install_dependencies():
    """Install dependencies untuk Termux"""
    print(f"{C.BLUE}🔧 Menginstal dependencies untuk Termux...{C.RESET}")
    
    # Untuk Termux, kita perlu install paket Python
    import subprocess
    import sys
    
    try:
        # Cek jika requests sudah terinstall
        import requests
        print(f"{C.GREEN}✅ requests sudah terinstall{C.RESET}")
    except ImportError:
        print(f"{C.YELLOW}📦 Menginstall requests...{C.RESET}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    
    print(f"{C.GREEN}✅ Semua dependencies siap!{C.RESET}")
    print(f"{C.CYAN}Jalankan: python elena.py{C.RESET}")


if __name__ == "__main__":
    # Cek jika ingin install dependencies
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_dependencies()
    else:
        try:
            main()
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}👋 ELENA AI dihentikan{C.RESET}")
        except Exception as e:
            print(f"{C.RED}❌ Fatal error: {str(e)}{C.RESET}")
            print(f"{C.YELLOW}Coba jalankan dengan: python elena.py --install{C.RESET}")
