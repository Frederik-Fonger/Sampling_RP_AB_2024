import run
from multiprocessing import Pool
from multiprocessing import Process, Queue
import random

# starts the program with multiple Event logs

sources_p_cc_all = ["data/Eventlogs/DomesticDeclarations/DomesticDeclarations.xes",
                 "data/Eventlogs/PermitLog/PermitLog.xes",
                 "data/Eventlogs/BPIC 2012/BPI_Challenge_2012.xes",
                 "data/Eventlogs/sepsis/Sepsis Cases - Event Log.xes",
                 "data/Eventlogs/InternationalDeclarations/InternationalDeclarations.xes",
                 "data/Eventlogs/PrepaidTravelCost/PrepaidTravelCost.xes",
                 "data/Eventlogs/RequestForPayment/RequestForPayment.xes"]



def rand_num(queue, string=""):
    run.run(string)
    queue.put(string)


if __name__ == "__main__":
    queue = Queue()

    processes = [Process(target=rand_num, args=(queue, x)) for x in sources_p_cc_all]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = [queue.get() for p in processes]

    print(results)
    print("All processes are finished")

