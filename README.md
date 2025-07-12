# WeaveHacks - Google Home AI Assistant

A custom Google Home integration that overrides normal behavior to call your `/prompt` API endpoint and read responses aloud.

## 🎯 Overview

This project allows you to:
- Override Google Home's default behavior
- Route all voice commands to your custom `/prompt` API
- Process commands through your own AI logic using OpenAI GPT
- Have Google Home speak the responses back to you

## 🏗️ Architecture

```
Google Home → Cloud Function → Flask API → OpenAI GPT → Response
```

1. **Google Home**: Captures voice commands
2. **Cloud Function**: Webhook that processes Google Home requests
3. **Flask API**: Your custom `/prompt` endpoint with OpenAI integration
4. **OpenAI GPT**: Processes queries and generates responses
5. **Response**: Google Home speaks the AI response

## 📁 Project Structure

```
WeaveHacks/
├── src/
│   ├── app.py                 # Flask API with /prompt endpoint
│   └── services/
│       ├── ping_gpt.py        # OpenAI GPT integration
│       ├── web_search.py      # Web search functionality
│       └── call.py            # Phone call functionality
├── index.js                   # Google Cloud Function webhook
├── package.json               # Cloud Function dependencies
├── smart-home-manifest.json   # Google Smart Home device definition
├── venv/                      # Python virtual environment
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Set up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

**Get your OpenAI API key:**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add it to your `.env` file

### 2. Start the Flask API

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install flask flask-cors openai python-dotenv

# Start the Flask app
cd src && python app.py
```

Your API will be available at: `http://localhost:5000/prompt`

### 3. Test the API

```bash
# Test the endpoint
curl -X POST http://localhost:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?"}'
```

Expected response:
```json
{
  "response": "Hello! I'm doing well, thank you for asking. How can I help you today?"
}
```

### 4. Deploy Cloud Function

```bash
# Navigate to project root
cd /path/to/WeaveHacks

# Deploy to Google Cloud Functions
gcloud functions deploy googleHomeWebhook \
  --runtime nodejs18 \
  --trigger-http \
  --allow-unauthenticated
```

### 5. Set up Google Smart Home Action

1. Go to [Actions on Google Console](https://console.actions.google.com/)
2. Create a new project
3. Enable Smart Home API
4. Choose "Cloud to Cloud" integration
5. Use the Cloud Function URL as your webhook
6. Configure device types using `smart-home-manifest.json`

## 🔧 Configuration

### Update API URL

In `index.js`, replace the API URL with your public endpoint:

```javascript
// For local testing
const apiResponse = await fetch('http://localhost:5000/prompt', {

// For production (replace with your public URL)
const apiResponse = await fetch('https://your-app.railway.app/prompt', {
```

### Custom Device Type

The project includes a custom device type with these traits:
- **OnOff**: Turn the AI assistant on/off
- **TextResponse**: Handle text-based responses
- **Modes**: Switch between chat and assist modes

## 🌐 Making Your API Public

### Option 1: ngrok (Quick Testing)

```bash
# Sign up at https://dashboard.ngrok.com/signup
# Get your authtoken and configure
ngrok config add-authtoken YOUR_TOKEN

# Create public tunnel
ngrok http 5000
```

### Option 2: Railway (Recommended)

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Deploy: `railway up`

### Option 3: Heroku

```bash
# Create Procfile
echo "web: gunicorn src.app:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 🧪 Testing

### Test Flask API Locally

```bash
# Test ping endpoint
curl http://localhost:5000/

# Test prompt endpoint
curl -X POST http://localhost:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather today?"}'
```

### Test Cloud Function

```bash
# Get your function URL
gcloud functions describe googleHomeWebhook --format="value(httpsTrigger.url)"

# Test the webhook
curl -X POST YOUR_FUNCTION_URL \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Google Home"}'
```

## 🔍 Troubleshooting

### Common Issues

1. **Flask app not starting**:
   ```bash
   # Make sure you're in the virtual environment
   source venv/bin/activate
   pip install flask flask-cors openai python-dotenv
   ```

2. **OpenAI API errors**:
   ```bash
   # Check your API key is set correctly
   cat .env
   
   # Test OpenAI connection
   python -c "from src.services.ping_gpt import gpt; print(gpt('Hello'))"
   ```

3. **Cloud Function deployment fails**:
   ```bash
   # Check if you're in the right directory
   ls package.json index.js
   
   # Make sure gcloud is configured
   gcloud config list
   ```

4. **API not accessible**:
   - Check if Flask is running on port 5000
   - Verify firewall settings
   - Use `host='0.0.0.0'` in Flask for external access

### Debug Mode

Enable debug logging in the Cloud Function by checking the logs:

```bash
gcloud functions logs read googleHomeWebhook
```

## 🔒 Security Considerations

- **Never commit your `.env` file** - it contains sensitive API keys
- The Cloud Function is set to `--allow-unauthenticated` for testing
- For production, implement proper authentication
- Consider rate limiting for your API
- Use HTTPS for all public endpoints

## 📝 Customization

### Modify AI Logic

Edit `src/app.py` to customize the AI behavior:

```python
@app.route("/prompt", methods=["POST"])
def prompt():
    # Extract query
    if request.is_json:
        data = request.get_json()
        user_query = data.get('query', '')
    else:
        user_query = request.form.get('query', '')
    
    # Customize the system prompt
    system_prompt = "You are a helpful AI assistant integrated with Google Home. Provide clear, concise, and helpful responses that would be appropriate for voice output."
    
    # Use different models or parameters
    ai_response = gpt(
        prompt=user_query,
        system=system_prompt,
        model="gpt-4.1-mini"  # or "gpt-4o-mini", "gpt-4o"
    )
    
    return jsonify({"response": ai_response})
```

### Add New Device Traits

Update `smart-home-manifest.json` to add new capabilities:

```json
{
  "traits": [
    "action.devices.traits.OnOff",
    "action.devices.traits.TextResponse",
    "action.devices.traits.Volume"  // Add new trait
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Google Cloud Function logs
3. Verify your Flask API is running
4. Ensure your public URL is accessible
5. Check your OpenAI API key is valid

## 🔗 Useful Links

- [Google Smart Home API Documentation](https://developers.google.com/assistant/smarthome)
- [Google Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Actions on Google Console](https://console.actions.google.com/)