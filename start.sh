#!/bin/bash

# Ensure the models directory exists
mkdir -p /code/app/clients/service/models

# Check if models need to be initialized
if [ ! -f "/code/app/clients/service/models/random_forest.pkl" ]; then
    echo "Initializing models..."
    cd /code
    # Run the model initialization function
    python -c "from app.clients.service.model import main; main()"
    echo "Models initialized successfully!"
fi

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload