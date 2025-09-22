# CV Matcher - Render Deployment Guide

## 🚀 Deploy to Render

### Prerequisites
1. GitHub repository with your CV Matcher code
2. Render account at [https://dashboard.render.com/](https://dashboard.render.com/)

### Step 1: Prepare Your Repository

Ensure your repository contains:
- `app.py` - Production Flask application
- `render.yaml` - Render configuration
- `requirements.txt` - Python dependencies
- `data/` directory with CVs and positions
- `templates/` directory with HTML templates

### Step 2: Deploy on Render

1. **Connect Repository**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   - **Name**: `cv-matcher`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Python Version**: `3.9.18`

3. **Environment Variables**
   - `PYTHON_VERSION`: `3.9.18`
   - `PORT`: `10000` (Render will override this)

4. **Advanced Settings**
   - **Health Check Path**: `/health`
   - **Disk**: Add persistent disk for CV data
     - Name: `cv-data`
     - Mount Path: `/opt/render/project/data`
     - Size: 1GB

### Step 3: Upload Data Files

After deployment, you'll need to upload your CV and position data:

1. **Via Render Shell** (Recommended)
   ```bash
   # Access your service shell in Render dashboard
   mkdir -p /opt/render/project/data/cv
   mkdir -p /opt/render/project/data/positions
   
   # Upload files using scp or git
   ```

2. **Via Git** (Alternative)
   - Add your data files to the repository
   - Push to trigger redeployment

### Step 4: Verify Deployment

1. **Health Check**: Visit `https://your-app.onrender.com/health`
2. **Main App**: Visit `https://your-app.onrender.com/`
3. **API Test**: Test the positions endpoint

### 🔧 Configuration Files

#### `render.yaml`
```yaml
services:
  - type: web
    name: cv-matcher
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.18
    healthCheckPath: /
    disk:
      name: cv-data
      mountPath: /opt/render/project/data
      sizeGB: 1
```

#### `app.py`
- Production-ready Flask application
- Health check endpoint
- Error handling
- Logging configuration

### 📊 Performance Considerations

- **Memory**: Render Starter plan has 512MB RAM
- **CPU**: Shared CPU resources
- **Storage**: 1GB persistent disk for CV data
- **Cold Starts**: May take 30-60 seconds on first request

### 🐛 Troubleshooting

1. **Build Failures**
   - Check `requirements.txt` for version conflicts
   - Ensure all dependencies are compatible

2. **Runtime Errors**
   - Check logs in Render dashboard
   - Verify data files are uploaded correctly

3. **Memory Issues**
   - Consider upgrading to higher plan
   - Optimize CV processing batch size

### 🔄 Updates

To update your deployment:
1. Push changes to your GitHub repository
2. Render will automatically redeploy
3. Check deployment logs for any issues

### 📈 Monitoring

- **Health**: `/health` endpoint
- **Logs**: Available in Render dashboard
- **Metrics**: CPU, memory, and response time

### 💰 Cost

- **Starter Plan**: $7/month
- **Disk Storage**: Included in plan
- **Bandwidth**: 100GB included

For production use, consider upgrading to higher plans for better performance.

