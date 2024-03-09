# 1.Given an array of numbers,find Second smallest and second largest elements.
# optimal approach:
import math

small = math.inf
secondSmall = math.inf
large = -math.inf
secLarge = -math.inf

nums = [1, 2, 4, 7, 7, 5]

for i in nums:
    if i > large:
        large, secLarge = i, large
    elif i > secLarge and i != large:
        secLarge = i

    if i < small:
        small, secondSmall = i, small
    elif i < secondSmall and i != secondSmall:
        secondSmall = i

print(secLarge)
print(secondSmall)

# ======================================================================

# 2.Given a sorted array having duplicates, re arrange the array in-place in a way that all unique elements,
# while preserving their order present at first.
# eg: nums = [1,1,2,2,2,3,3,4,4,4]; After rearrangement: [1,2,3,4,numbers in any order]


# Brute:
# Make a traversal and keep adding the elements to an ordered set. As the set doesn't store repeated elements,
# just repopulate all the elements in the ordered set in the starting indices of the array.
# M: Number of unique elements
# N: All the given elements.
# TC: while populating the set :O(N) + while repopulating: O(M)
# SC: O(M) (For set)
#
# Code:
#

s = set()

for i in nums:
    s.add(i)
nums[:len(s)] = [j for j in s]


# Optimal:

# This uses a two pointer approach:

# My solution:
def removeDupicates(nums):
    i = 1
    j = 2
    while i <= len(nums) - 1 and j <= len(nums) - 1:
        # while nums[i - 1] <= nums[i]:
        #     i += 1
        # j = i + 1
        while nums[i - 1] >= nums[j]:
            j += 1
            if j > len(nums) - 1:
                # nums = nums[:i]
                return i
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j = i + 1


# Leetcode Solution:
# Trace this to get an idea.
"""
Move the j pointer from index 1(potential idx having chances of a duplicate) and check for the element that's
different from i pointer (that refers to the unique elements). if I find any, just update the (i + 1)th index with j
and then increase i the pointer by 1. loop ends when j reaches the len(nums)-1.
"""


def removeDuplicates(nums):
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            nums[i + 1] = nums[j]
            i += 1
    return nums


# ====================================================================================
#
# Move zeros to the end.
# Pb: You are given an array of integers, your task is to move
# all the zeros in the array to the end of the array and move non-negative integers to
# the front by maintaining their order.
#
# arr = [1, 0, 2, 3, 0, 4, 0, 1]
# o/p:  [1, 2, 3, 4, 1, 0, 0, 0]
#
# Optimal:

def moveZeroesToEnd(nums):
    i = 0
    # Find the first zero.
    while nums[i] != 0:
        i += 1

    j = i + 1
    while j <= len(nums) - 1:
        if nums[j] != 0:
            # Swap the current zero with the first non-zero element that we come across while traversal.
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
        j += 1
    return nums


# ========================================================================================================
# Problem Statement: Given two sorted arrays, arr1, and arr2 of size n and m. Find the union of two sorted arrays.
#
# The union of two arrays can be defined as the common and distinct elements in the
# two arrays.NOTE: Elements in the union should be in ascending order.


def union(a, b):
    merged = []
    i = 0
    j = 0

    # Loop until the lengths of the two lists are same.
    while i <= len(a) - 1 and j <= len(b) - 1:
        # If ele from first list is less,
        if a[i] < b[j]:
            merged.append(a[i])
            i += 1
        # If eles are equal.
        elif a[i] == b[j]:
            merged.append(a[i])
            i += 1
            j += 1
        # If ele from second list is greater,
        else:
            merged.append(b[j])
            j += 1
    """Note: Only one of the below gets executed and the other won't as the condition doesn't satisfies."""
    # Loop to append any leftover eles from list a
    while i <= len(a) - 1:
        merged.append(a[i])
        i += 1
    # Loop to append any leftover eles from list b
    while j <= len(b) - 1:
        merged.append(b[j])
        j += 1
    return merged


# ====================================================================================
# Problem Statement: Given an array that contains only 1 and 0
# return the count of maximum consecutive ones in the array.
#
# lst = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
#
# Optimal:
def maxConsecutiveOnes(arr):
    i = 0
    # Variable to be updated with maximum count of consecutive 1s
    maxCount = 0
    # Variable to count the consecutive 1s of a particular group.
    clusterCnt = 0
    while i < len(arr):
        # If we see 1, increase the (current) clusterCount by one.
        if arr[i] == 1:
            clusterCnt += 1
        # If we see a 0, it means we might have just came out of a group so, update the maxCount
        else:
            # We re-initialize the clusterCnt to zero and update maxCount only if we
            # have a cluster at hand whose num of consecutive ones is greater than the 
            # one we are already having.
            if clusterCnt > maxCount:
                maxCount = clusterCnt
                clusterCnt = 0

        i += 1
    return maxCount


# ========================================================================================
#
# Problem Statement: Given a non-empty array of integers arr,
# every element appears twice except for one. Find that single one.
#
# Optimal: (XOR)
#
# Properties of xor (^):
# Xor of any number with itself is 0 i.e, n ^ n = 0
# Xor of a number with zero is number itself i.e., n ^ 0 = n

def singleOccuringEle(arr):
    xor = 0
    for ele in arr:
        xor ^= ele
    return xor


lst = [7, 1, 2, 1, 2]
ans = singleOccuringEle(lst)
print(ans)


# ====================================================================================================
#
# Problem: Given an array and a sum k, we need to print the length of the longest subarray that sums to k.
#
# arr = [4, 1, 2, 3, 4]
# k = 5

# The following code works even if both positives and negatives are present in the array.
# Better approach for maximum subarray whose sum equals given target.
def longestSubArrayWithGivenSum(arr, target):
    # This won't get reset to zero
    sum_ = 0
    maxlen = 0
    # The dictionary to store the sum of all elements to it's left of the current index.
    untilNowSums = {}
    for i in range(len(arr)):
        # Firstly, curSum cur ele to the sum-tracking variable(sum_) as the iteration proceeds
        sum_ += arr[i]

        # If the sum from starting to until now (which is stored in sum_) equals target, then we have the
        # biggest subarray as of current iteration.
        if sum_ == target:
            maxlen = i + 1

        # If sum until is either lesser or greater than the target we are looking for,
        # this 'if' statement gets activated.
        rem = sum_ - target
        if rem in untilNowSums:
            maxlen = max(maxlen, i - untilNowSums[rem])

        # Storing the sum until now (ith idx)
        if sum_ not in untilNowSums:
            untilNowSums[sum_] = i
    # After loop terminates, return the maxLen.
    return maxlen


# Optimal:
# Two pointers.
# Reason why this doesn't work when arr contains positives and negatives:
# observe that when the sum exceed the target, we are decreasing the range of [left:right] by
# moving the left pointer towards right pointer. i.e, there lies a belief that the sum definitely bound to
# change in decreasing direction. but if negatives are present in between, sum fluctuates which is counter to
# our belief.(Think of sum as a linear graph, it's slope is -45 degrees as per our belief(sum decreases if left pointer
# moves towards right pointer).
# But, it is a graph that will contain sharp trough and crests if there are positives and negatives.)
def getLongestSubarray(a: [int], k: int) -> int:
    n = len(a)  # size of the array.

    left, right = 0, 0  # 2 pointers
    Sum = a[0]
    maxLen = 0
    while right < n:
        # if sum > k, reduce the subarray from left
        # until sum becomes less or equal to k:
        while left <= right and Sum > k:
            Sum -= a[left]
            left += 1

        # if sum = k, update the maxLen i.e. answer:
        if Sum == k:
            maxLen = max(maxLen, right - left + 1)

        # Move forward the right pointer:
        right += 1
        if right < n: Sum += a[right]

    return maxLen


# =====================================================================================
# Problem Statement: Given an array of integers arr[] and an integer target.
# Optimal:
def twoSum(arr, tar):
    # Store the ele:indexPosition as we traverse through the array.
    # (eleRef-elementReference)
    eleRef = {}
    for i in range(len(arr)):
        val = tar - arr[i]
        # If the value exists in the eleRef, return the current ele and that ele.
        if val in eleRef:
            return [i, eleRef[val]]
        # If the current element is not added into the eleRef yet, curSum it.
        if arr[i] not in eleRef:
            eleRef[arr[i]] = i


lst = [2, 7, 11, 15]
target = 9
ans = twoSum(lst, target)
print(ans)
# ===========================================================================

# Dutch National Flag algorithm
# Problem Statement: Given an array consisting of only 0s, 1s, and 2s.
# Write a program to in-place sort the array without using inbuilt sort functions.
# (Expected: Single pass-O(N) and constant space)
# Brute: Merge sort the array. TC: O(N*LogN)

# Trace with the examples below to get a grip.
lst2 = [1, 2, 2, 0, 1, 0, 1]
lst = [1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def dnfBrute(arr):
    low = 0
    mid = 0
    high = len(arr) - 1
    while mid <= high:
        if arr[mid] == 0:
            arr[mid], arr[low] = arr[low], arr[mid]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            # mid += 1
            high -= 1
    return arr


ans = dnfBrute(lst)
print(ans)

# ============================================================================================

# Problem Statement: Given an array of N integers, write a program to return an element
# that occurs more than N/2 times in the given array. You may consider that such an
# element always exists in the array.

nums = [4, 4, 7, 4, 12, 4, 4, 6]


# This works when Qn states that given array definitely contains a majority element.
# i.e, the element that occurs more than n/2 times in an array.
# If the question doesn't states so, we have to manually check
# if we have really got the correct answer(As shown in last loop).

def mooresVotingAlgo(arr):
    ele = arr[0]
    counter = 1
    for i in range(1, len(arr)):
        if arr[i] == ele:
            counter += 1
        else:
            counter -= 1
        if counter == 0:
            if i <= len(arr) - 2:
                ele = arr[i + 1]
                counter = 1
    # Checking if it is a correct answer
    majorityCount = 0
    for j in arr:
        if j == ele:
            majorityCount += 1
    if majorityCount > len(arr) / 2:
        return ele
    # When the given array doesn't contain majority element.
    return -1


ans = mooresVotingAlgo(nums)
print(ans)

# ==============================================================================

# Problem Statement: Given an integer array arr, find the contiguous subarray
# (containing at least one number) which has the largest sum and returns its
# sum and prints the subarray(starting and ending positions).
import math


def kadanesAlgorithm(nums):
    sum_ = 0
    maxSum = -math.inf
    startOfSubArray = None
    endOfSubArray = None
    for k in range(len(nums)):
        sum_ += nums[k]

        # We just discard the sum if it goes below 0 and start summing up from the next element.
        if sum_ < 0:
            sum_ = 0
            startOfSubArray = k + 1

        if sum_ > maxSum:
            maxSum = sum_
            endOfSubArray = k
    # Return the starting,ending and maximum subarray sum
    return [startOfSubArray, endOfSubArray, maxSum]


arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
ans = kadanesAlgorithm(arr)
print(ans)

# ===========================================================================

# stock buy and sell problem
# Problem Statement: You are given an array of prices where prices[i] is the price of a given stock on an ith day.
#
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day
# in the future to sell that stock. Return the maximum profit you can achieve from this transaction. If
# you cannot achieve any profit, return 0.

# LeetCode Two Pointer approach:
# Do a dry on the following examples
stockPricesOnIthDay = [7, 1, 6, 5, 0, 9, 20]


# stockPricesOnIthDay = [3, 2, 6, 5, 0, 3]
# stockPricesOnIthDay = [1, 2]
def maxProfit(self, arr) -> int:
    left = 0
    right = 1
    profit = 0
    while left <= right <= len(arr) - 1:
        if arr[left] >= arr[right]:
            left = right
            right += 1
        else:
            profit = max(profit, arr[right] - arr[left])
            right += 1
    return profit


# The solution that I tried to implement(but couldn't at that time.)
# This is actually more intuitive.
def maxProfit(prices, price=None) -> int:
    profit = 0

    # buy opportunity
    # buy until next is bigger

    # From there, calculate different buying options, until a smallest number is found
    # Keeping track of current max profit

    boughtPrice = math.inf

    for price in prices:
        # When I can buy the stock for (lower)'price' itself rather than (Higher)'boughtPrice'.
        if price < boughtPrice:
            boughtPrice = price
        else:
            # If the price is greater than the price I bought, choose the profit that I can get at best.
            profit = max(profit, price - boughtPrice)

    return profit


# ===========================================================================

# Problem Statement:
#
# There’s an array ‘A’ of size ‘N’ with an equal number of positive and negative elements. Without altering the
# relative order of positive and negative elements, you must return an array of alternately positive and negative
# values.
#
# Note: Start the array with positive elements.
# Optimal: Runs in O(N) TC
nums = [1, 2, -3, -1, -2, 3]
rearrangedArr = [0] * len(nums)
posIdx = 0
negIdx = 1
for i in range(len(nums)):
    if nums[i] > 0:
        rearrangedArr[posIdx] = nums[i]
        posIdx += 2
    else:
        rearrangedArr[negIdx] = nums[i]
        negIdx += 2
print(rearrangedArr)

# =========================================================================

# pb Statement:Given an unsorted array of integers nums, return the length of
# the longest consecutive elements sequence.
# Optimal solution.
def longestConsecutive(nums):
    # To eliminate duplicates
    hashSet = set(nums)
    # When the nums is an empty list, we can return 0 as the answer.
    longest = 0
    for n in hashSet:
        # Check if the current element is the starting of the sequence.
        if n - 1 not in hashSet:
            length = 1
            # This is like checking for 2,3,4 etc., by 1 + 1,1 + 2,1 + 3 etc., when we are at
            # 1 as the starting point
            # Do a dry run with [100,200,1,4,3,2] array
            while n + length in hashSet:
                length += 1
            longest = max(longest, length)
    return longest


# Problem Statement: Given a matrix if an element in the matrix is 0 then you will have to
# set its entire column and row to 0 and then return the matrix.
matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
# Brute
m = len(matrix)
n = len(matrix[0])
zeroLocs = []
for i in range(m):
    for j in range(n):
        if matrix[i][j] == 0:
            zeroLocs.append((i, j))

for zero in zeroLocs:
    row = zero[0]
    col = zero[1]
    for c in range(n):
        matrix[row][c] = 0
    for r in range(m):
        matrix[r][col] = 0

print(matrix)
# =========================================================================


# matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
# Optimal :(In terms of space complexity O(1))
"""
Approach:
First traverse the matrix to update the row and col headers to zero when matrix[i][j] = 0.
This indicates that these rows and cols (wherever we marked zeros),will be zeros
Now, Traverse teh matrix by leaving the headers.i.e, from 1,1 to m-1,n-1
and make the element itself zero if it is non-zero iff either its row header/Col header is zero.
Finally, Header row and col are left to be dealt with,
I have taken the following config.
(suppose, shape of matrix is 4,3)
  zeroCol
    * $ $ $
    *
    *
    *
Now, I will start with columns (Refer striver's video to know why.)
and update them if the 0,0 is zero.
Then, I will start with row header updating.
if the zeroCol is zero, then all eles are updated as zero.

At the end, return the matrix
"""

matrix = [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]
# Num of rows
m = len(matrix)
# Num of columns
n = len(matrix[0])
zeroCol = 1
for i in range(m):
    for j in range(n):
        # If the element is zero,
        if matrix[i][j] == 0:
            # Update the row
            matrix[i][0] = 0

            # Update the column
            if j > 0:
                matrix[0][j] = 0
            else:
                # when it is zeroth column
                zeroCol = 0

# Iterate rows and cols except header row and col
for rInner in range(1, m):
    for cInner in range(1, n):
        # If the val at current Iter is non-zero and either of its headers is zero, mark the element as zero
        if matrix[rInner][cInner] != 0 and (matrix[rInner][0] == 0 or matrix[0][cInner] == 0):
            matrix[rInner][cInner] = 0

# Dealing with Col Header
if matrix[0][0] == 0:
    for k in range(n):
        matrix[0][k] = 0

# Dealing with row Header
if zeroCol == 0:
    for l in range(m):
        matrix[l][0] = 0
print(matrix)

# ==============================================================
# pb: Rotate the matrix by 90 degrees
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# After 90 degrees rotation: [[7,4,1],[8,5,2],[9,6,3]]
m = len(matrix)
n = len(matrix[0])
print(matrix)
for i in range(m):
    for j in range(n):
        if i < j:
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
for row in range(m):
    matrix[row].reverse()
print(matrix)

# ==================================================================
# pb: print matrix in spiral order

# matrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# print matrix in spiral order
m = len(matrix)
n = len(matrix[0])

leftExtreme = 0
rightExtreme = n - 1
topExtreme = leftExtreme + 1
botExtreme = m - 1

while leftExtreme <= rightExtreme and topExtreme <= botExtreme:
    i = j = leftExtreme
    # j = leftExtreme
    while i != leftExtreme + 1 or j != leftExtreme:
        # -->
        while j <= rightExtreme:
            print(matrix[i][j], end=" ")
            j += 1
        j -= 1
        i += 1

        # top to bottom
        while i <= botExtreme:
            print(matrix[i][j], end=" ")
            i += 1
        i -= 1
        j -= 1

        # <--
        while j >= leftExtreme:
            print(matrix[i][j], end=" ")
            j -= 1
        j += 1
        i -= 1

        # bottom to top
        while i >= topExtreme:
            print(matrix[i][j], end=" ")
            i -= 1
        i += 1
    leftExtreme += 1
    rightExtreme -= 1
    topExtreme = leftExtreme + 1
    botExtreme -= 1

while leftExtreme <= rightExtreme:
    print(matrix[m // 2][leftExtreme], end=" ")
    leftExtreme += 1


# Problem Statement: Given an array of integers and an integer k,
# return the total number of subarrays whose sum equals k.

arr = [1, 2, 3, -3, 1, 1, 1, 4, 2, -3]
# Optimal:
subArrays = 0
k = 3
# we start with the belowDictionary having 1 key value pair -> 0 occurred : 1 time
SumK_CntV = {0: 1}
add = 0
for i in range(len(arr)):
    add += arr[i]

    # Check if we have came across our desired value in the dictionary
    if add - k in SumK_CntV:
        subArrays += SumK_CntV[add - k]

    # if the add is not in dictionary, add it. else, increase the count (I.e, no. of times the add occurred until now.)
    if add not in SumK_CntV:
        SumK_CntV[add] = 1
    else:
        SumK_CntV[add] += 1

print(subArrays)

# Better: Two pointers to dynamically evaluate if sum == k; TC: O(N**2)
i = 0
while i <= len(arr) - 2:
    j = i + 1
    add = arr[i]
    if add == k:
        subArrays += 1
    while j <= len(arr) - 1:
        add += arr[j]
        if add == k:
            subArrays += 1
        j += 1
    i += 1
print(subArrays)

# Brute force: Generate all sub arrays and check if their sum equals k
