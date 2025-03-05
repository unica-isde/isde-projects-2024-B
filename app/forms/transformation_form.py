from fastapi import Request


class TransformationForm:
    def __init__(self, request: Request) -> None:
        self.request: Request = request
        self.errors: list = []
        self.image_id: str = ""
        self.color: float = 0.0
        self.brightness: float = 0.0
        self.contrast: float = 0.0
        self.sharpness: float = 0.0


    async def load_data(self):
        form = await self.request.form()
        self.image_id = form.get("image_id")
        self.model_id = form.get("model_id")
        self.color = float(form.get("color"))
        self.brightness = float(form.get("brightness"))
        self.contrast = float(form.get("contrast"))
        self.sharpness = float(form.get("sharpness"))

    def is_valid(self):
        if not self.image_id or not isinstance(self.image_id, str):
            self.errors.append("A valid image id is required")
        if not self.errors:
            return True
        return False