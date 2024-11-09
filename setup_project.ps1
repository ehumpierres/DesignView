# setup_project.ps1

param(
    [string]$projectName = "designview"
)

Write-Host "Creating project structure for $projectName..." -ForegroundColor Green

# Create main directories
$directories = @(
    "app",
    "app/models",
    "app/services",
    "app/routes",
    "app/utils"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir
    Write-Host "Created directory: $dir" -ForegroundColor Yellow
}

# Create Python files
$pythonFiles = @(
    "app/__init__.py",
    "app/main.py",
    "app/models/__init__.py",
    "app/models/product.py",
    "app/services/__init__.py",
    "app/services/search_engine.py",
    "app/services/s3_handler.py",
    "app/routes/__init__.py",
    "app/routes/api.py",
    "app/utils/__init__.py",
    "app/utils/helpers.py"
)

foreach ($file in $pythonFiles) {
    New-Item -ItemType File -Force -Path $file
    Write-Host "Created Python file: $file" -ForegroundColor Yellow
}

# Create config files
$configFiles = @(
    "requirements.txt",
    "Procfile",
    "runtime.txt",
    ".env",
    ".gitignore",
    "config.py",
    "wsgi.py",
    "README.md"
)

foreach ($file in $configFiles) {
    New-Item -ItemType File -Force -Path $file
    Write-Host "Created config file: $file" -ForegroundColor Yellow
}

# Create and activate virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv
Write-Host "Virtual environment created" -ForegroundColor Yellow

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate
Write-Host "Virtual environment activated" -ForegroundColor Yellow

# Initialize git repository
Write-Host "Initializing git repository..." -ForegroundColor Green
git init
Write-Host "Git repository initialized" -ForegroundColor Yellow

Write-Host "Project setup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Add your dependencies to requirements.txt" -ForegroundColor White
Write-Host "2. Run: pip install -r requirements.txt" -ForegroundColor White
Write-Host "3. Configure your .env file" -ForegroundColor White
Write-Host "4. Create your Heroku app: heroku create $projectName" -ForegroundColor White