from pydantic import BaseModel, Field
class Member(BaseModel):
    # gt=0 &lt;<
    name:str = Field(min_length=2, max_length=10, description="이름")
    id:int   = Field(gt=0)