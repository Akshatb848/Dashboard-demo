#!/bin/bash

# 🔧 One-Click Deployment Fix Script
# Fixes Python 3.13 incompatibility issues

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║          🔧 STREAMLIT CLOUD DEPLOYMENT FIX SCRIPT                ║"
echo "║                                                                  ║"
echo "║   This will fix Python 3.13 incompatibility in your repo        ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if we're in a git repo
if [ ! -d .git ]; then
    echo -e "${RED}❌ Error: Not a git repository${NC}"
    echo "Please run this script from your repository root directory"
    exit 1
fi

echo -e "${BLUE}📋 Creating fix files...${NC}"
echo ""

# 1. Update requirements.txt
echo -e "${BLUE}1. Updating requirements.txt...${NC}"
cat > requirements.txt << 'EOF'
streamlit>=1.40.0
pandas>=2.2.0
numpy>=1.26.0,<2.0.0
plotly>=5.24.0
prophet>=1.1.5
scipy>=1.13.0
openpyxl>=3.1.5
xlrd>=2.0.1
python-dateutil>=2.9.0
pytz>=2024.1
EOF
echo -e "${GREEN}✅ requirements.txt updated${NC}"
echo ""

# 2. Create .python-version
echo -e "${BLUE}2. Creating .python-version...${NC}"
echo "3.11" > .python-version
echo -e "${GREEN}✅ .python-version created (Python 3.11)${NC}"
echo ""

# 3. Create packages.txt
echo -e "${BLUE}3. Creating packages.txt...${NC}"
cat > packages.txt << 'EOF'
build-essential
gfortran
EOF
echo -e "${GREEN}✅ packages.txt created${NC}"
echo ""

# 4. Show changes
echo -e "${BLUE}📝 Changes to be committed:${NC}"
git status --short
echo ""

# 5. Confirm
read -p "Do you want to commit and push these fixes? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}🚀 Committing changes...${NC}"
    
    git add requirements.txt .python-version packages.txt
    git commit -m "Fix: Python 3.11 compatibility - Updated dependencies for Streamlit Cloud"
    
    echo -e "${GREEN}✅ Changes committed${NC}"
    echo ""
    
    echo -e "${BLUE}📤 Pushing to remote...${NC}"
    git push
    
    echo -e "${GREEN}✅ Pushed to GitHub${NC}"
    echo ""
    
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║                     🎉 FIX APPLIED! 🎉                           ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo ""
    echo "1. Go to Streamlit Cloud dashboard"
    echo "2. Your app will auto-redeploy (takes 2-3 minutes)"
    echo "3. Watch the logs for successful deployment"
    echo ""
    echo -e "${GREEN}Expected success message:${NC}"
    echo '  "📦 Processing dependencies... ✅"'
    echo '  "🎉 App is live!"'
    echo ""
    echo -e "${YELLOW}If deployment still fails, check:${NC}"
    echo "  • DEPLOYMENT_FIX.md for detailed troubleshooting"
    echo "  • Streamlit Cloud logs for specific errors"
    echo ""
else
    echo ""
    echo -e "${YELLOW}⚠️  Changes NOT committed${NC}"
    echo "You can review the files and commit manually:"
    echo "  git add requirements.txt .python-version packages.txt"
    echo "  git commit -m 'Fix: Python compatibility'"
    echo "  git push"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Script completed. Good luck with deployment! 🚀         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
