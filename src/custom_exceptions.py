class WeatherAPIError(Exception):
    """
    Exception raised for errors in the Weather API interaction.
    
    Attributes:
        message -- explanation of the error
        status_code -- HTTP status code returned by the API (if applicable)
    """
    def __init__(self, message="Error occurred with the Weather API", status_code=None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self):
        if self.status_code:
            return f"{self.message} (Status Code: {self.status_code})"
        return self.message


class ModelLoadError(Exception):
    """
    Exception raised when a machine learning model fails to load.
    
    Attributes:
        model_name -- name of the model that failed to load
        message -- explanation of the error
    """
    def __init__(self, model_name, message="Failed to load the model"):
        self.model_name = model_name
        self.message = f"{message}: {self.model_name}"
        super().__init__(self.message)

    def __str__(self):
        return self.message