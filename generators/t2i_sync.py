import io
import json

import numpy as np
import requests
import torch
from PIL import Image

# Import the base class
from ..core.base import WanAPIBase


class WanT2ISyncGenerator(WanAPIBase):
    """Node for text-to-image synchronous generation using Wan model"""

    # Define available Wan models
    MODEL_OPTIONS = [
        "z-image-turbo",
        "qwen-image-max",
        "wan2.6-t2i",
    ]

    # Define allowed sizes for Wan models with descriptive names
    # Based on the documentation, Wan supports sizes from 512 to 1440 pixels
    SIZE_OPTIONS = [
        "1024*1024",  # 1:1 square (default)
        "1152*896",   # 9:7 landscape
        "896*1152",   # 7:9 portrait
        "1280*720",   # 16:9 landscape
        "720*1280",   # 9:16 portrait
        "1440*512",   # Wide landscape
        "512*1440"    # Tall portrait
        "768*768"     # 1:1 square
        "1440*1440"   # 1:1 square
    ]

    # Define region options
    REGION_OPTIONS = [
        "international",
        "mainland_china"
    ]

    def __init__(self):
        super().__init__()
        self.model = "wan2.6-t2i"  # Using Wan Speed Edition as default

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODEL_OPTIONS, {
                    "default": "wan2.6-t2i"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate an image of a cat"
                }),
                "size": (cls.SIZE_OPTIONS, {
                    "default": "1024*1024"
                }),
                "region": (cls.REGION_OPTIONS, {
                    "default": "international"
                })
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "prompt_extend": ("BOOLEAN", {
                    "default": True
                }),
                "watermark": ("BOOLEAN", {
                    "default": False
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")  # Returns image tensor and image URL
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "generate"
    CATEGORY = "Ru4ls/Wan"

    def generate(self, model, prompt, size, region, negative_prompt="", prompt_extend=True, watermark=False, seed=0):
        # Check API key based on region
        api_key = self.check_api_key(region)

        # Get the appropriate API endpoints based on region
        endpoints = self.get_api_endpoints(region)
        api_url = endpoints["t2i_post_sync"]

        # Set the selected model
        self.model = model

        # Debug: Print API key status
        print(f"Using API key: {api_key[:8]}...{api_key[-4:] if api_key else 'None'}")
        print(f"Selected model: {self.model}")
        print(f"Using API endpoint: {api_url}")
        print(f"Selected region: {region}")

        # Prepare API payload for text-to-image generation - using the Wan format
        payload = {
            "model": self.model,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{
                        "text": prompt
                    }]
                }]
            },
            "parameters": {
                "size": size,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
                "n": 1  # Generate only one image
            }
        }

        # Add optional parameters if they have non-default values
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        if seed > 0:
            payload["parameters"]["seed"] = seed

        # Set headers according to DashScope documentation
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-DataInspection": '{"input": "disable", "output": "disable"}'
        }

        # Debug: Print request details
        debug_headers = {**headers, "Authorization": f"Bearer {api_key[:8]}..."}
        print(f"Request headers: {debug_headers}")
        print(f"Request payload:\n{json.dumps(payload, indent=2)}")

        try:
            # Make API request
            print(f"Making API request to {api_url}")
            response = requests.post(api_url, headers=headers, json=payload)
            print(f"Response status code: {response.status_code}")
            if hasattr(response, 'text'):
                print(f"Response text: {response.text[:500]}...")  # Print first 500 chars
            response.raise_for_status()

            result = response.json()
            # Print first 200 chars
            print(f"API response received: {json.dumps(result, indent=2)[:200]}...")

            # Check if this is a task creation response
            if "output" not in result or "choices" not in result["output"]:
                raise ValueError(f"Unexpected API response format: {result}")

            choices = result["output"]["choices"]
            if (len(choices) == 0 or "message" not in choices[0]
                or "content" not in choices[0]["message"]
                or len(choices[0]["message"]["content"]) == 0
                    or "image" not in choices[0]["message"]["content"][0]):
                raise ValueError(f"Unexpected API response format: {result}")

            task_id = result["request_id"]
            task_status = choices[0]["finish_reason"]
            print(f"Task created with ID: {task_id}, status: {task_status}")

            if task_status != "stop":
                raise ValueError(f"Unexpected task status: {task_status}")

            # Task completed successfully
            image_url = choices[0]["message"]["content"][0]["image"]

            # Download the generated image
            image_response = requests.get(image_url)
            image_response.raise_for_status()

            # Convert to tensor
            image = Image.open(io.BytesIO(image_response.content))
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Return both the image tensor and the image URL
            return (image_tensor, image_url)

        except requests.exceptions.RequestException as e:
            # More detailed error handling
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                response_text = e.response.text
                print(f"API request failed with status {status_code}: {response_text}")
                if status_code == 401:
                    raise RuntimeError(f"API request failed: 401 Unauthorized. "
                                       f"This usually means your API key is invalid or not properly configured. "
                                       f"Error details: {response_text}")
                elif status_code == 403:
                    raise RuntimeError(f"API request failed: 403 Forbidden. "
                                       f"This usually means your API key is valid but you don't have access to this model. "
                                       f"Error details: {response_text}")
                elif status_code == 400:
                    raise RuntimeError(f"API request failed: 400 Bad Request. "
                                       f"This usually means there's an issue with the request format. "
                                       f"Error details: {response_text}")
                else:
                    raise RuntimeError(
                        f"API request failed: {status_code} {e.response.reason}. Response: {response_text}")
            else:
                raise RuntimeError(f"API request failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to process API response: {str(e)}")
