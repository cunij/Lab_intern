import subprocess
import sys

# 기존 루프(스케일 스윕) 잠시 비활성화
# subprocess.run([sys.executable, "main.py", "--scalar","1", "--image_number", "0" ], check=True)
# subprocess.run([sys.executable, "evaluation.py" ], check=True)
# image=1
# for i in range(50,30,-3):
#     scale=i/100
#     
#     subprocess.run([sys.executable, "main.py", "--scalar", str(scale),"--image_number", str(scale), "--save-files", "1", "--plot", "1"], check=True)
#     subprocess.run([sys.executable, "evaluation.py" ], check=True)
#     print("=================")
#     print()
#     image+=1

# # congestion source=0.5로 뽑고, scalar=1에 적용

# for i in range(10,1,-1):
#     for j in range(5,0,-1):
#         scale=i/100
#         cost=j/10
#         subprocess.run(
#             [
#                 sys.executable,
#                 "main.py",
#                 "--scalar",
#                 "1",
#                 "--congestion-reweight",
#                 "1",
#                 "--congestion-source-scalar",
#                 "0.5",
#                 "--congestion-cost-alpha",
#                 str(scale),
#                 "--congestion-cost-cap",
#                 str(cost),
                
#                 "--plot",
#                 "1",
#                 "--save-files",
#                 "1",
#             ],
#             check=True,
#         )
#         subprocess.run([sys.executable, "evaluation.py"], check=True)
# for n in range(40,60):
#     d=n/100
#     print(f"====== {d} ========\n")
#     subprocess.run(
#                 [
#                     sys.executable,
#                     "main.py",
#                     "--scalar",
#                     "1",
#                     "--congestion-reweight",
#                     "1",
#                     "--congestion-source-scalar",
#                     str(d),
#                     "--congestion-cost-alpha",
#                     "0.03",
#                     "--congestion-cost-cap",
#                     "0.1",
#                     "--congestion-top-n",
#                     "7",
#                     "--plot",
#                     "1",
#                     "--save-files",
#                     "1",
#                 ],
#                 check=True,
#             )
#     subprocess.run([sys.executable, "evaluation.py"], check=True)
#     print()
#     print()

subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--scalar",
                    "1",
                    "--congestion-reweight",
                    "0",
                    "--congestion-source-scalar",
                    "0.5",
                    "--congestion-cost-alpha",
                    "0.03",
                    "--congestion-cost-cap",
                    "0.2",#별 상관 없음
                    "--congestion-top-n",
                    "7",
                    "--plot",
                    "1",
                    "--save-files",
                    "1",
                ],
                check=True,
            )
print()
subprocess.run([sys.executable, "evaluation.py"], check=True)
print()
print()