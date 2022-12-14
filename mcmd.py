# Maximum Centre-Disjoint Mergeable Disks
#
# Copyright (C) 2022 Ali Gholami Rudi <ali at rudi dot ir>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
import math
import sys
import pulp

disks = []		# input disks: centre + radius
sigma = []		# sigma[i]: sorted points based on distance to disk[i]
rigma = []		# reverse of sigma
rads = []		# disk radii
cens = []		# disk centres
dist_haver = False	# use haversine formula for distance

def haversine(a, b):
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
	n = len(disks)
	cens.extend([d[0] for d in disks])
	rads.extend([d[1] for d in disks])
	sigma.extend([None] * len(disks))
	for i in range(len(disks)):
		sigma[i] = list(range(len(disks)))
		sigma[i].sort(key=lambda p: dist(cens[i], cens[p]))
	for i in range(len(disks)):
		rad = 0
		for j in range(len(sigma[i])):
			if j > 0 and rad < dist(cens[i], cens[sigma[i][j]]):
				del sigma[i][j:]
				break
			rad += rads[sigma[i][j]]
	rigma.extend([list() for i in range(len(disks))])
	for i in range(len(disks)):
		for j in sigma[i]:
			rigma[j].append(i)

def linprog():
	prob = pulp.LpProblem("MergeableDisks", pulp.LpMinimize)
	vars_list = [(j, i) for i in range(len(disks)) for j in sigma[i]]
	# x[i, j] indicates that disk i is merged with disk j
	x = pulp.LpVariable.dicts("x", vars_list, 0, 1, cat="Integer")
	# Each disk is merged at most once
	for i in range(len(disks)):
		prob += pulp.lpSum(x[i, j] for j in rigma[i]) == 1
	# A can be merged with B if all closer disks to B are also merged with it
	for i in range(len(disks)):
		for j in range(len(sigma[i]) - 1):
			prob += x[sigma[i][j], i] >= x[sigma[i][j + 1], i]
	# Disks must be centre-disjoint
	for i in range(len(disks)):
		for j in range(len(disks)):
			if i != j:
				d = dist(cens[i], cens[j]) + 1000000 * (1 - x[j, j])
				prob += pulp.lpSum(x[k, i] * rads[k] for k in sigma[i]) <= d
	# Maximize selected disks
	prob += pulp.lpSum(-x[i, i] for i in range(len(disks)))
	prob.solve()
	phi = [-1] * len(disks)
	for i in range(len(disks)):
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
	for i in range(len(disks)):
		if phi[i] != i:
			print(f"  {i} -> {phi[i]}")
	outs = [0 for i in range(len(disks))]
	for i in range(len(disks)):
		outs[phi[i]] += rads[i]
	print("Resulting disks:")
	for i in range(len(disks)):
		if outs[i]:
			print(f"  {cens[i][0]} {cens[i][1]}  {outs[i]}")
