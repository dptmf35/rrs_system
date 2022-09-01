import time
import uvicorn
import warnings
warnings.filterwarnings('ignore')
from get_reciprocal_score import ReciprocalScore
from get_optimal_weight import WeightOptimizer


from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse

app = FastAPI()
rpc = ReciprocalScore()
opt = WeightOptimizer()


class inputMember(BaseModel) :
    memNo : int


@app.post('/get_rcf')
async def main(input_member : inputMember) :
    start = time.time()
    result = {}

    try :
        mem_no = input_member.memNo
        # a = opt.find_optimal_weight(mem_no)
        # top_k, explanation = rpc.get_reciprocal_score(mem_no, k=1)
        top_k, explanation = rpc.test(mem_no, k=1)
        # print('\n top_k :',top_k[0])
        # print('explanation :',explanation[0])
        result = {'top_k' : top_k[0], 'explanation' : explanation[0]}
    except Exception as e:
        print(e)
        pass
    finally :
        print(f'elapsed time : {time.time() - start:.3f}')
        return JSONResponse(result)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

# if __name__ == '__main__' :
#     start = time.time()
#     print('=' * 40)
#     x = 1697018
#     a = opt.find_optimal_weight(x)
#     top_k, explanation = rpc.get_reciprocal_score(x, a=a, k=1)
#     print(top_k, explanation)
#     print('=' * 40)
#     print(f'elapsed time : {time.time() - start}')