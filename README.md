# X Automation Bot with Local LLM

A Python bot that automatically interacts with X posts using a small local LLM that runs on your GPU. The bot can like posts and generate contextual replies using AI.

## Features

- 🤖 **Local LLM Integration**: Uses small, GPU-optimized models like DialoGPT-small
- 🎯 **Smart Tweet Filtering**: Filters tweets based on keywords and content quality
- 🔄 **Automated Interactions**: Likes and replies to tweets with AI-generated responses
- ⚡ **GPU Acceleration**: Optimized for GPU inference with fallback to CPU
- 🛡️ **Rate Limiting**: Built-in delays and limits to avoid detection
- 🎨 **Manual Login**: You log in manually to X, bot takes over automation
- 📊 **Logging & Monitoring**: Comprehensive logging of all activities

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (recommended) or CPU
- Chrome browser
- 4GB+ RAM (8GB+ recommended for GPU)

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Settings**
   ```bash
   copy .env.example .env
   # Edit .env file with your preferences
   ```

3. **Run the Bot**
   ```bash
   python x_bot.py
   ```

5. **Manual Login**
   - The bot will open Chrome and navigate to X
   - Log in manually with your credentials
   - Press Enter when you're logged in and see your timeline
   - Bot takes over from there!

## Configuration

Edit the `.env` file to customize behavior:

```env
# How many tweets to interact with per session
MAX_TWEETS_PER_SESSION=10

# Delay between interactions (seconds)
MIN_DELAY_SECONDS=30
MAX_DELAY_SECONDS=120

# Keywords to look for in tweets
KEYWORDS=AI,technology,programming,machine learning

# LLM model to use
LLM_MODEL=microsoft/DialoGPT-small

# How often to run automation (minutes)
AUTOMATION_INTERVAL_MINUTES=30

# Interaction probabilities (0.0 to 1.0)
LIKE_PROBABILITY=0.8
REPLY_PROBABILITY=0.3
```

## Supported Models

The bot supports several small, GPU-friendly models:

- **microsoft/DialoGPT-small** (Default) - ~117M parameters
- **microsoft/DialoGPT-medium** - ~354M parameters  ***Seems to work the best
- **facebook/blenderbot_small-90M** - ~90M parameters
- **google/flan-t5-small** - ~80M parameters

## How It Works

1. **Manual Login**: You log in to X manually in the browser
2. **Post Discovery**: Bot scrolls through your timeline to find relevant posts
3. **Content Filtering**: Filters posts based on keywords and quality metrics
4. **AI Response Generation**: Uses local LLM to generate contextual replies
5. **Automated Interaction**: Likes posts (80% chance) and posts AI-generated replies (30% chance)
6. **Rate Limiting**: Implements delays and limits to avoid detection

## Safety Features

- **Rate Limiting**: Configurable delays between actions
- **Daily Limits**: Maximum interactions per day
- **Content Filtering**: Avoids spam and promotional content
- **Duplicate Prevention**: Tracks processed tweets
- **Manual Oversight**: You control the login process

## GPU Requirements

For optimal performance:
- **NVIDIA GPU**: GTX 1060 / RTX 2060 or better
- **VRAM**: 4GB+ (6GB+ recommended)
- **CUDA**: Version 11.0+ installed

The bot will automatically detect and use your GPU if available, falling back to CPU if needed.

## Troubleshooting

### Model Loading Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Selenium Issues
- Ensure Chrome browser is installed
- Check if ChromeDriver downloads automatically
- Try running with `HEADLESS_MODE=True` in .env

### Memory Issues
- Use smaller models (DialoGPT-small vs medium)
- Reduce `MAX_TWEETS_PER_SESSION`
- Close other applications to free RAM

## Project Structure

```
linkedin_impersonator/
├── x_bot.py              # Main bot implementation  
├── requirements.txt     # Python dependencies
├── .env.example        # Configuration template
├── .env                # Your configuration (create this)
├── x_bot.log           # Bot activity logs
└── README.md          # This file
```

## Legal & Ethics

⚠️ **Important**: 
- Use responsibly and follow X's Terms of Service
- Don't spam or harass users
- Respect rate limits and community guidelines
- This is for educational/personal use only

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes. Use at your own risk and responsibility.