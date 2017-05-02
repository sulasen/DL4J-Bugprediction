package Bugprediction.Iterators;

import com.github.javaparser.ast.Node;
import com.github.javaparser.utils.Pair;

import java.util.*;

/**
 * Created by Sebi on 1/16/2017.
 */
public class TreeIterator {
    private Node next;

    public TreeIterator(Node root){
        next = root;
        if(next == null)
            return;
        while (next.getChildNodes().get(0) != null)
            next = next.getChildNodes().get(0);
    }

    public boolean hasNext(){
        return next != null;
    }

    public Node next(){
        if(!hasNext()) throw new NoSuchElementException();
        Node r = next;
        List<Node> children = next.getChildNodes();
        // if you can walk right, walk right, then fully left.
        // otherwise, walk up until you come from left.
        if(children.size() > 0 && next.getChildNodes().get(children.size()-1) != null){
            next = next.getChildNodes().get(children.size()-1);
            while (next.getChildNodes().get(0) != null)
                next = next.getChildNodes().get(0);
            return r;
        }else while(true){
            if(next.getParentNode() == null){
                next = null;
                return r;
            }
            if(next.getParentNode().get().getChildNodes().get(0) == next){
                next = next.getParentNode().get();
                return r;
            }
            next = next.getParentNode().get();
        }
    }

    public int getDistance(Node node1, Node node2){
        List<Pair<Node, Integer>> parentNodes1 = new LinkedList<Pair<Node, Integer>>();
        List<Pair<Node, Integer>> parentNodes2 = new LinkedList<Pair<Node, Integer>>();
        int count1 = 0;
        int count2 = 0;
        parentNodes1.add(new Pair(node1, 0));
        parentNodes2.add(new Pair(node2, 0));
        while(node1.getParentNode().isPresent()){
            count1++;
            node1 = node1.getParentNode().get();
            parentNodes1.add(new Pair(node1, count1));
        }
        while(node2.getParentNode().isPresent()){
            count2++;
            node2 = node2.getParentNode().get();
            parentNodes2.add(new Pair(node1, count2));
        }

        return 1;
    }
}
