import argparse
from configparser import ConfigParser
import flowMASAC 

cfg = ConfigParser()
cfg.read("config.ini")

actor_lr = float(cfg["train_parameters"]["actor_lr"])
critic_lr = float(cfg["train_parameters"]["critic_lr"])
num_episodes = float(cfg["train_parameters"]["num_episodes"])

# get local iteration inforamtion
parser = argparse.ArgumentParser()

parser.add_argument("-tau", type=float, required=False, default=0.001, help="soft update rate")
parser.add_argument("-batch_size", type=int, required=False, default=256, help="batch size number")
parser.add_argument("-comm_interval", type=int, required=False, default=8, help="communication interval")
parser.add_argument("-test_interval", type=int, required=False, default=10, help="test interval")
parser.add_argument("-repeat", type=int, required=False, default=1, help="test numbers")
parser.add_argument("-cuda", type=int, required=False, default=0, help="gpu idx")
parser.add_argument("-seg", type=int, required=False, default=4, help="segment number")
parser.add_argument("-re", type=int, required=False, default=3, help="replica number")
parser.add_argument("-maxstep", type=int, required=False, default=10, help="maxstep")
parser.add_argument("-radius", type=int, required=False, default=90, help="communication range")
parser.add_argument("-output", type=str, required=False, default = None, help="filepath of outputs")
parser.add_argument("-fine_tuning", type=bool, required=False, default = False, help="whether load the converged model to fine tuning in new environments")
parser.add_argument("-checkpoint", type=bool, required=False, default = False, help="whether save the final model")
parser.add_argument("-loadfile", type=str, required=False, default = None, help="load converged model file")
parser.add_argument("-loss_rate", type=float, required=False, default = 0, help="Communication loss rate.")
parser.add_argument("-testOnly", type=bool, required=False, default = False, help="whether only test, not train the policy.")


args = parser.parse_args()

tau = args.tau
batch_size = args.batch_size
comm_interval = args.comm_interval
test_interval = args.test_interval
repeat = args.repeat
cuda = args.cuda
segments = args.seg
replica = args.re
max_step = args.maxstep
radius = args.radius
head = args.output
fine_tuning = args.fine_tuning
checkpoint = args.checkpoint
loadfile = args.loadfile
testOnly = args.testOnly
loss_rate = args.loss_rate


flowMASAC.train(actor_lr, 
                critic_lr, 
                num_episodes, 
                tau, 
                batch_size, 
                comm_interval, 
                test_interval, 
                repeat, 
                cuda, 
                segments, 
                replica, 
                max_step, 
                radius, 
                head, 
                fine_tuning, 
                checkpoint,
                loadfile,
                loss_rate,
                testOnly)