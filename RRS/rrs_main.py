from get_RRS import ReciprocalRecommenderSystem
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse
import traceback
# from utils.config.logger import LogConfig
import time
import os
import uvicorn

# log = LogConfig()
app = FastAPI()
pid = os.getpid()
rrs = ReciprocalRecommenderSystem()

class getMemNo(BaseModel) :
    memNo : int

@app.post('/recom')
async def reciprocalrecommender(i : getMemNo) :
    request_time = time.time()
    result = {}
    try :
        # log.Log(f"[API Request] pid : {pid} - memNo : {i.memNo}")
        res, exp = rrs.get_RRS(i.memNo, top_k=1)
        result['scoreList'] = res
        result['explanations'] = exp
    except Exception as e :
        trace_back = traceback.format_exc()
        print(trace_back)
        # log.error_log(f"[ERROR] [Failed Reuest - memNo : {i.memNo} \n {trace_back}")

    finally:
        result['responseTime'] = round(time.time() - request_time, 2)
        return JSONResponse(result)

# uvicorn
if __name__ == '__main__' :
    uvicorn.run('rrs_main:app', host='0.0.0.0', port=8031, access_log=False,
        reload=True)
