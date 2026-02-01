# 🚀 Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Testing](#local-testing)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Railway Deployment](#railway-deployment)
6. [AWS EC2 Deployment](#aws-ec2-deployment)
7. [Production Checklist](#production-checklist)

---

## Prerequisites

- Python 3.9+
- Git installed
- GitHub account
- (Optional) Docker installed
- (Optional) Cloud platform account

---

## Local Testing

### Step 1: Clone Repository

```bash
git clone https://github.com/Akshatb848/AI-Analytics-Dashboard.git
cd AI-Analytics-Dashboard
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Locally

```bash
streamlit run app_production.py
```

Your app will open at `http://localhost:8501`

### Testing Checklist

- [ ] Upload CSV file successfully
- [ ] Semantic profiler runs without errors
- [ ] KPIs are discovered and ranked
- [ ] Forecasting works (if time column exists)
- [ ] Insights are generated
- [ ] Charts render correctly
- [ ] No console errors

---

## Streamlit Cloud Deployment

**Best for:** Free hosting, automatic updates, easy setup

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - AI Analytics Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/AI-Analytics-Dashboard.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Authenticate with GitHub
4. Select your repository
5. **Main file path**: `app_production.py`
6. **Python version**: 3.9
7. Click "Deploy!"

### Step 3: Wait for Deployment

- Build time: 2-5 minutes
- Watch logs for any errors
- Once deployed, you'll get a URL like:
  ```
  https://yourusername-ai-analytics-dashboard.streamlit.app
  ```

### Automatic Updates

- Every push to `main` branch auto-deploys
- Monitor logs in Streamlit Cloud dashboard

---

## Docker Deployment

**Best for:** Self-hosting, consistent environments

### Build Image

```bash
docker build -t ai-analytics-dashboard .
```

### Run Container

```bash
docker run -p 8501:8501 ai-analytics-dashboard
```

Access at `http://localhost:8501`

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    restart: unless-stopped
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
    volumes:
      - ./data:/app/data  # Optional: mount data directory
```

Run with:
```bash
docker-compose up -d
```

---

## Railway Deployment

**Best for:** Free tier, easy scaling, automatic HTTPS

### Method 1: GitHub Integration

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Select your repository
6. Railway auto-detects and deploys
7. Get your URL: `https://your-app.railway.app`

### Method 2: Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

### Environment Variables (if needed)

```bash
railway variables set PORT=8501
```

---

## AWS EC2 Deployment

**Best for:** Full control, enterprise use

### Step 1: Launch EC2 Instance

- **AMI**: Ubuntu Server 22.04 LTS
- **Instance Type**: t2.medium (2 vCPU, 4GB RAM)
- **Security Group**: Open ports 22 (SSH), 80 (HTTP), 8501 (Streamlit)

### Step 2: Connect and Setup

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.9
sudo apt install python3.9 python3.9-venv python3-pip -y

# Clone repository
git clone https://github.com/YOUR_USERNAME/AI-Analytics-Dashboard.git
cd AI-Analytics-Dashboard

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run with systemd (Production)

Create service file:

```bash
sudo nano /etc/systemd/system/streamlit-dashboard.service
```

Add content:

```ini
[Unit]
Description=AI Analytics Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/AI-Analytics-Dashboard
Environment="PATH=/home/ubuntu/AI-Analytics-Dashboard/venv/bin"
ExecStart=/home/ubuntu/AI-Analytics-Dashboard/venv/bin/streamlit run app_production.py --server.port=8501 --server.address=0.0.0.0

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit-dashboard
sudo systemctl start streamlit-dashboard

# Check status
sudo systemctl status streamlit-dashboard
```

### Step 4: Setup Nginx (Optional)

```bash
sudo apt install nginx -y

sudo nano /etc/nginx/sites-available/streamlit
```

Add configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 5: SSL with Let's Encrypt (Optional)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## Production Checklist

### Security

- [ ] Change default ports
- [ ] Enable HTTPS/SSL
- [ ] Set up firewall rules
- [ ] Use environment variables for secrets
- [ ] Implement rate limiting
- [ ] Add authentication (if needed)

### Performance

- [ ] Enable caching
- [ ] Optimize Prophet parameters
- [ ] Use appropriate instance size
- [ ] Monitor memory usage
- [ ] Set up auto-scaling (if needed)

### Monitoring

- [ ] Set up logging
- [ ] Configure error tracking
- [ ] Monitor uptime
- [ ] Track user metrics
- [ ] Set up alerts

### Backup

- [ ] Regular code backups (GitHub)
- [ ] Database backups (if applicable)
- [ ] Configuration backups

### Documentation

- [ ] User guide
- [ ] API documentation (if applicable)
- [ ] Deployment notes
- [ ] Troubleshooting guide

---

## Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| **Streamlit Cloud** | ✅ 3 apps | $20/mo | Quick demos |
| **Railway** | ✅ $5 credit | Usage-based | Side projects |
| **Heroku** | ❌ Removed | $7/mo+ | Legacy apps |
| **AWS EC2** | ✅ 12 months | $10+/mo | Enterprise |
| **DigitalOcean** | ❌ No | $4/mo+ | Self-hosting |
| **Docker** | ✅ Free | Hosting cost | Full control |

---

## Troubleshooting

### Issue: Prophet installation fails

**Solution:**
```bash
# Install system dependencies
sudo apt-get install python3-dev
pip install pystan==2.19.1.1
pip install prophet
```

### Issue: Out of memory

**Solution:**
- Use smaller instance
- Implement data sampling
- Add pagination
- Optimize Prophet parameters

### Issue: Slow loading

**Solution:**
- Enable Streamlit caching: `@st.cache_data`
- Use data sampling for large files
- Optimize database queries

### Issue: Port already in use

**Solution:**
```bash
# Find process
lsof -ti:8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run app_production.py --server.port=8502
```

---

## Updates and Maintenance

### Update Application

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service (systemd)
sudo systemctl restart streamlit-dashboard

# Or restart Docker
docker-compose restart
```

### Monitor Logs

```bash
# Streamlit Cloud: Use dashboard

# systemd:
sudo journalctl -u streamlit-dashboard -f

# Docker:
docker logs -f container_name
```

---

## Support

For deployment issues:
1. Check logs first
2. Consult platform documentation
3. Open GitHub issue with details
4. Include error messages and environment info

---

**Deployment completed successfully? Don't forget to:**
- ⭐ Star the repository
- 📢 Share with your team
- 📝 Document any customizations
- 🎉 Celebrate!
