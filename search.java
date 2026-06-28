import java.util.Arrays;
public class search{
    static boolean search2(String str, char target){
        if(str.length()==0){
            return false;
        }
        for(char ch: str.toCharArray()){
            if(ch== target){
                return true;
            }
        }
        return false;
    }
    public static void main(String args[]){
        String name="Rudrakshi";
        char target='h';
        System.out.println(Arrays.toString(name.toCharArray()));
    }
} 