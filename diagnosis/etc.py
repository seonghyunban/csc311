class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        i = 0
        j = len(s) - 1

        ss = s.lower()

        while (i < j):
            while (i < j) and not (ss[i].isalnum() and ss[j].isalnum()):
                if not ss[i].isalnum():
                    i += 1
                if not ss[j].isalnum():
                    j -= 1
            # we know that (i >= j) or (s[i].isalpha() and s[j].isalpha())
            if (i >= j):
                return True
            elif ss[i] != ss[j]:
                return False
            else:
                i += 1
                j -= 1

        return True






def isPalindrome(s):
    """
    :type s: str
    :rtype: bool
    """
    i = 0
    j = len(s) - 1

    ss = s.lower()

    while (i < j):
        while (i < j) and not (s[i].isalnum() and s[j].isalnum()):
            if not s[i].isalnum():
                i += 1
            if not s[j].isalnum():
                j -= 1
        # we know that (i >= j) or (s[i].isalpha() and s[j].isalpha())
        print(s[i], s[j])
        if (i >= j):
            return True
        elif s[i] != s[j]:
            return False
        else:
            i += 1
            j -= 1

    return True

test = "A man, a plan, a canal: Panama"
isPalindrome(test)



