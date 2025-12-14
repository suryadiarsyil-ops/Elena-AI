
## 📁 **File elena_install.sh** (untuk Termux):

```bash
#!/data/data/com.termux/files/usr/bin/bash
# Install script untuk Termux

echo "🔧 Installing ELENA AI for Termux..."

# Update packages
pkg update -y && pkg upgrade -y

# Install Python dan tools
pkg install -y python git wget

# Install Python packages
pip install requests

# Clone repository
if [ -d "elena-ai" ]; then
    echo "📁 Repository sudah ada, melakukan update..."
    cd elena-ai
    git pull
else
    echo "📥 Cloning repository..."
    git clone https://github.com/suryadiarsyil-ops/Elena-AI.git
    cd elena-ai
fi

# Buat file executable
chmod +x elena.py

echo ""
echo "✅ Instalasi selesai!"
echo ""
echo "📝 Cara menggunakan:"
echo "1. cd elena-ai"
echo "2. python elena.py"
echo "3. Ketik '/setup' untuk setup API key pertama kali"
echo ""
echo "🔥 ELENA AI siap digunakan di Termux!"
