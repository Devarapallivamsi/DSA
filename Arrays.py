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
        # Firstly, add cur ele to the sum-tracking variable(sum_) as the iteration proceeds
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
        # If the current element is not added into the eleRef yet, add it.
        if arr[i] not in eleRef:
            eleRef[arr[i]] = i


lst = [2, 7, 11, 15]
target = 9
ans = twoSum(lst, target)
print(ans)


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
