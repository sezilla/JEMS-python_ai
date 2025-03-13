import os
from openai import OpenAI

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
)


response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """You are an AI scheduler for task categories. The categories are:
            - 1 Year to 6 Months before
            - 9 Months to 6 Months before
            - 6 Months to 3 Months before
            - 4 Months to 3 Months before
            - 3 Months to 1 Month before
            - 1 Month to 1 Week before
            - 1 Week before and Wedding Day
            - Wedding Day
            - 6 Months after Wedding Day
            
            Input format:
            - project_id: int
            - start: string (date)
            - end: string (date)

            Output format:
            - project_id: int
            - start: string (date)
            - end: string (date)
            - duration: int (span of project in days)
            - category_schedule: list (categories with start and deadline)
            """
        },
        {
            "role": "user",
            "content": """{
                "project_id": 123,
                "start": "2025-06-01",
                "end": "2026-06-01"
            }"""
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1
)

print(response.choices[0].message.content)
