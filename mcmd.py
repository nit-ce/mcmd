# Maximum Centre-Disjoint Mergeable Disks
import sys
import pulp

disks = []		# input disks: centre + radius
sigma = []		# sigma[i]: sorted points based on distance to disk[i]
rigma = []		# reverse of sigma
rads = []		# disk radii
cens = []		# disk centres
n = 0			# number of disks
dist_haver = False	# use haversine formula for distance

def haversine(a, b):
	import math
	R = 6371230;
	a1 = math.radians(a[0])
	o1 = math.radians(a[1])
	a2 = math.radians(b[0])
	o2 = math.radians(b[1])
	v = pow(math.sin((a2 - a1) / 2), 2) + \
		math.cos(a1) * math.cos(a2) * pow(math.sin((o2 - o1) / 2), 2)
	return abs(R * 2 * math.atan2(math.sqrt(v), math.sqrt(1 - v)))

def dist(a, b):
	if dist_haver:
		return haversine(a, b)
	return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def init_arrays():
	global n
	global sigma
	global rigma
	global rads
	global cens
	n = len(disks)
	cens = [d[0] for d in disks]
	rads = [d[1] for d in disks]
	sigma = [None] * n
	for i in range(n):
		sigma[i] = list(range(len(disks)))
		sigma[i].sort(key=lambda p: dist(cens[i], cens[p]))
	for i in range(n):
		rad = 0
		for j in range(len(sigma[i])):
			if j > 0 and rad < dist(cens[i], cens[sigma[i][j]]):
				del sigma[i][j:]
				break
			rad += rads[sigma[i][j]]
	rigma = [list() for i in range(n)]
	for i in range(n):
		for j in sigma[i]:
			rigma[j].append(i)

def linprog():
	prob = pulp.LpProblem("MergeableDisks", pulp.LpMinimize)
	vars_list = [(j, i) for i in range(n) for j in sigma[i]]
	# x[i, j] indicates that disk i is merged with disk j
	x = pulp.LpVariable.dicts("x", vars_list, 0, 1, cat="Integer")
	# Each disk is merged at most once
	for i in range(n):
		prob += pulp.lpSum(x[i, j] for j in rigma[i]) == 1
	# A can be merged with B if all closer disks to B are also merged with it
	for i in range(n):
		for j in range(len(sigma[i]) - 1):
			prob += x[sigma[i][j], i] >= x[sigma[i][j + 1], i]
	# Disks must be centre-disjoint
	for i in range(n):
		for j in range(n):
			if i != j:
				d = dist(cens[i], cens[j]) + 1000000 * (1 - x[j, j])
				prob += pulp.lpSum(x[k, i] * rads[k] for k in sigma[i]) <= d
	# Maximize selected disks
	prob += pulp.lpSum(-x[i, i] for i in range(n))
	prob.solve()
	phi = [-1] * n
	for i in range(n):
		for j in sigma[i]:
			if x[j, i].value():
				phi[j] = i
	return phi

def dataset():
	for l in sys.stdin:
		toks = [s.strip() for s in l.split()]
		if len(toks) == 3:
			disks.append(((float(toks[0]), float(toks[1])), float(toks[2])))

if __name__ == "__main__":
	for o in sys.argv[1:]:
		if o[:2] == '-v':
			dist_haver = True
	dataset()
	print(f"Disks: {len(disks)}")
	init_arrays()
	phi = linprog()
	print("Merges:")
	for i in range(n):
		if phi[i] != i:
			print(f"  {i} -> {phi[i]}")
	outs = [0 for i in range(n)]
	for i in range(n):
		outs[phi[i]] += rads[i]
	print("Resulting disks:")
	for i in range(n):
		if outs[i]:
			print(f"  {cens[i][0]} {cens[i][1]}  {outs[i]}")
